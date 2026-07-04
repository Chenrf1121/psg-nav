"""
Landmark Manager - Handles landmark filtering and frontier-related operations.

This module extracts landmark filtering, frontier detection, and landmark-frontier
proximity calculations from the main PSG_Nav_Agent class.
"""

import numpy as np
import torch


class LandmarkManager:
    """Manages landmark filtering, frontier detection, and proximity calculations."""

    def __init__(self, agent):
        """
        Initialize LandmarkManager.

        Args:
            agent: Reference to the main PSG_Nav_Agent instance
        """
        self.agent = agent
        self.map_resolution = agent.map_resolution
        self.map_height = agent.map_height

    def filter_landmarks(self, landmark_nodes, agent_position, min_distance=20, trajectory_exclusion_radius=31):
        """
        Filter landmarks to ensure minimum distance constraints and avoid visited areas.

        Args:
            landmark_nodes: Array of landmark positions [[row, col], ...]
            agent_position: Tuple (agent_row, agent_col) in map coordinates
            min_distance: Minimum distance from agent in pixels
            trajectory_exclusion_radius: Radius around landmark to check for visited areas

        Returns:
            numpy.ndarray: Filtered landmark nodes, or None if no landmarks pass the filter
        """
        if landmark_nodes is None or len(landmark_nodes) == 0:
            return None

        agent_pos = np.array([agent_position[0], agent_position[1]])  # [row, col] format

        # Step 1: Filter out landmarks too close to agent
        filtered_landmarks = landmark_nodes
        # Step 2: Filter out landmarks in visited areas (using agent_trajectory_map)
        if hasattr(self.agent, 'agent_trajectory_map') and self.agent.agent_trajectory_map is not None:
            h, w = self.agent.agent_trajectory_map.shape
            unvisited_landmarks = []

            for lm in filtered_landmarks:
                lm_r, lm_c = int(self.map_height - lm[0]), int(lm[1])

                # Define the region around this landmark
                r_min = max(0, lm_r - trajectory_exclusion_radius)
                r_max = min(h, lm_r + trajectory_exclusion_radius + 1)
                c_min = max(0, lm_c - trajectory_exclusion_radius)
                c_max = min(w, lm_c + trajectory_exclusion_radius + 1)

                # Extract the region around the landmark
                region = self.agent.agent_trajectory_map[r_min:r_max, c_min:c_max]

                # Check if ANY cell in this region has been visited (value != 0)
                if np.sum(region) == 0:
                    # All cells in the region are 0 (unvisited), keep this landmark
                    unvisited_landmarks.append(lm)
                # else: landmark is in/near visited area, exclude it

            filtered_landmarks = unvisited_landmarks

            if len(filtered_landmarks) == 0:
                return None

        # Step 3: Filter landmarks that are too close to each other (greedy algorithm)
        selected_landmarks = []

        # Sort by distance to agent (process closer landmarks first for better coverage)
        landmarks_with_dist = []
        for lm in filtered_landmarks:
            lm_pos = np.array([self.map_height - lm[0], lm[1]])
            dist = np.linalg.norm(lm_pos - agent_pos)
            landmarks_with_dist.append((lm, dist))
        landmarks_with_dist.sort(key=lambda x: x[1])

        # Greedy selection: add landmark if it's far enough from all selected landmarks
        for lm, _ in landmarks_with_dist:
            lm_pos = np.array([self.map_height - lm[0], lm[1]])
            is_far_enough = True

            for selected_lm in selected_landmarks:
                selected_pos = np.array([self.map_height - selected_lm[0], selected_lm[1]])
                dist = np.linalg.norm(lm_pos - selected_pos)
                if dist < min_distance:
                    is_far_enough = False
                    break

            if is_far_enough:
                selected_landmarks.append(lm)

        # Return filtered landmarks as numpy array
        return np.array(selected_landmarks) if len(selected_landmarks) > 0 else None

    def calculate_frontier_proximity(self, landmark_nodes, frontier_locations, threshold=50.0):
        """
        Calculate whether each landmark is near frontiers.

        Args:
            landmark_nodes: Array of landmark positions [[row, col], ...]
            frontier_locations: Tensor or array of frontier positions
            threshold: Distance threshold in pixels (default: 50.0)

        Returns:
            dict: Mapping of (row, col) -> frontier proximity info
        """
        landmark_near_frontier = {}

        if landmark_nodes is None or len(landmark_nodes) == 0:
            return landmark_near_frontier

        if frontier_locations is None:
            return landmark_near_frontier

        # Convert frontier_locations to numpy if it's a tensor
        if torch.is_tensor(frontier_locations):
            fl_np = frontier_locations.cpu().numpy()
        else:
            fl_np = frontier_locations

        for i, lm in enumerate(landmark_nodes):
            row, col = int(lm[0]), int(lm[1])
            lm_pos = np.array([row, col])

            # Calculate distance to nearest frontier
            min_dist_to_frontier = float('inf')
            for frontier_pos in fl_np:
                dist = np.linalg.norm(lm_pos - frontier_pos)
                if dist < min_dist_to_frontier:
                    min_dist_to_frontier = dist

            # Determine if near frontier
            is_near_frontier = min_dist_to_frontier <= threshold
            dist_to_frontier_meters = min_dist_to_frontier * self.map_resolution / 100.0

            # Store information using position as key
            landmark_key = (row, col)
            landmark_near_frontier[landmark_key] = {
                'near_frontier': is_near_frontier,
                'distance_to_frontier': float(min_dist_to_frontier),
                'distance_to_frontier_meters': float(dist_to_frontier_meters)
            }

            print(f"Landmark {i} ({row},{col}): {'Near' if is_near_frontier else 'Far from'} frontier, dist={dist_to_frontier_meters:.2f}m")

        return landmark_near_frontier

    def find_all_valid_frontier_midpoints(self, frontier_locations, trajectory_exclusion_radius=31,
                                           agent_position=None, min_distance_from_agent=20, min_cluster_size=3,
                                           min_linearity_ratio=0.6, min_distance_between_midpoints=10, fbe_free_map=None):
        """
        Find all valid frontier cluster midpoints, excluding visited areas and points too close to agent.
        Uses dynamic linearity requirements based on cluster size: larger clusters need lower linearity.
        Only keeps frontiers reachable from agent's current region (not just the largest region).

        Args:
            frontier_locations: Tensor or numpy array of frontier points, each point is [row, col]
            trajectory_exclusion_radius: Radius around midpoint to check for visited areas (in pixels)
            agent_position: Tuple (agent_row, agent_col) in map coordinates. If None, no distance check is performed.
            min_distance_from_agent: Minimum distance from agent in pixels (default 20 = 1 meter at 5cm resolution)
            min_cluster_size: Minimum number of continuous frontier points to form a valid cluster (default 3)
            min_linearity_ratio: DEPRECATED - now uses dynamic linearity based on cluster size
                                Dynamic strategy:
                                - Small clusters (3 points): require high linearity (0.80)
                                - Medium clusters (5-10 points): require medium linearity (0.70-0.77)
                                - Large clusters (>20 points): accept lower linearity (0.40)
                                - Linearity requirement decreases by 0.035 for every 2 additional points
            min_distance_between_midpoints: Minimum distance between any two midpoints in pixels (default 10 = 0.5m at 5cm resolution)
            fbe_free_map: Free space map for connectivity check. If provided, only keeps frontiers reachable from agent's region.

        Returns:
            numpy array of valid landmarks [[row, col], ...] representing all valid midpoints,
            or None if no valid frontiers available
        """
        if frontier_locations is None or len(frontier_locations) == 0:
            return None

        # Convert to numpy if tensor
        if torch.is_tensor(frontier_locations):
            fl_np = frontier_locations.cpu().numpy()
        else:
            fl_np = frontier_locations

        # Find agent's connected component in fbe_free_map - only keep frontiers reachable from agent
        # This is better than using "largest region" because agent might be in a smaller region
        agent_region_mask = None
        if fbe_free_map is not None and agent_position is not None:
            from scipy import ndimage
            # Convert to numpy if tensor
            if torch.is_tensor(fbe_free_map):
                fbe_free_np = fbe_free_map.cpu().numpy()
            else:
                fbe_free_np = fbe_free_map

            # Ensure 2D array (squeeze extra dimensions)
            if fbe_free_np.ndim > 2:
                fbe_free_np = fbe_free_np.squeeze()
                if fbe_free_np.ndim > 2:
                    # If still more than 2D, take first channel
                    fbe_free_np = fbe_free_np[..., 0]

            # Label connected components in free space (value == 1)
            labeled_array, num_features = ndimage.label(fbe_free_np == 1)
            if num_features > 0:
                # Find which component the agent is in
                agent_row, agent_col = int(agent_position[0]), int(agent_position[1])
                h, w = labeled_array.shape

                if 0 <= agent_row < h and 0 <= agent_col < w:
                    agent_component_label = labeled_array[agent_row, agent_col]

                    if agent_component_label > 0:  # Agent is in free space
                        agent_region_mask = (labeled_array == agent_component_label)
                        component_sizes = np.bincount(labeled_array.ravel())
                        print(f"[Frontier Connectivity] Found {num_features} free space regions, "
                              f"agent is in region with {component_sizes[agent_component_label]} pixels")
                    else:
                        # Agent is not in free space (shouldn't happen, but handle gracefully)
                        print(f"[Frontier Connectivity] Warning: Agent position {agent_position} is not in free space")
                else:
                    print(f"[Frontier Connectivity] Warning: Agent position {agent_position} is out of bounds")

        from sklearn.cluster import DBSCAN

        # Use DBSCAN to cluster frontier points into continuous lines
        clustering = DBSCAN(eps=2, min_samples=2).fit(fl_np)
        labels = clustering.labels_

        # Find all clusters (excluding noise labeled as -1)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if len(unique_labels) == 0:
            # No valid clusters
            print(f"[Frontier] No frontier clusters found")
            return None

        # Pre-compute distance transform if we have agent_region_mask
        # This avoids recomputing it for every cluster
        distance_map = None
        if agent_region_mask is not None:
            from scipy.ndimage import distance_transform_edt
            # Invert the mask so agent_region = 0, other areas = 1
            inverted_mask = (~agent_region_mask).astype(np.uint8)
            distance_map = distance_transform_edt(inverted_mask)

        # Calculate the midpoint of each cluster and check if it's valid
        valid_midpoints = []
        filtered_size = 0  # Count clusters filtered due to size
        filtered_linearity = 0  # Count clusters filtered due to non-linearity
        filtered_distance = 0  # Count clusters filtered due to proximity to existing midpoints
        filtered_connectivity = 0  # Count clusters filtered due to connectivity

        for label in unique_labels:
            cluster_points = fl_np[labels == label]

            # Filter 1: only keep clusters with at least min_cluster_size points
            if len(cluster_points) < min_cluster_size:
                filtered_size += 1
                continue

            # Sort points to form a continuous path (greedy nearest neighbor)
            sorted_points = [cluster_points[0]]
            remaining = list(cluster_points[1:])

            while remaining:
                last_point = sorted_points[-1]
                distances = [np.linalg.norm(last_point - p) for p in remaining]
                nearest_idx = np.argmin(distances)
                sorted_points.append(remaining[nearest_idx])
                remaining.pop(nearest_idx)

            sorted_points = np.array(sorted_points)

            # Calculate endpoint distance (straight line distance between first and last point)
            endpoint_distance = np.linalg.norm(sorted_points[-1] - sorted_points[0])

            # Calculate total path length (sum of consecutive point distances)
            path_length = np.sum(np.linalg.norm(sorted_points[1:] - sorted_points[:-1], axis=1))

            # Calculate linearity ratio
            # If cluster forms a straight line: linearity_ratio ≈ 1.0
            # If cluster forms a circle or curve: linearity_ratio < 0.5
            linearity_ratio = endpoint_distance / path_length if path_length > 0 else 0

            # Filter 2: Dynamic linearity threshold based on cluster size
            # Strategy: Larger clusters can have lower linearity requirements
            # Small clusters (3 points) need high linearity (0.80)
            # Large clusters (>20 points) can accept lower linearity (0.40)
            cluster_size = len(cluster_points)

            # Calculate required linearity dynamically
            # Start with high requirement (0.80) for min_cluster_size (3 points)
            # Decrease by 0.035 for every 2 additional points
            # Bottom out at 0.40 for very large clusters
            max_required_linearity = 0.80  # For smallest valid clusters (3 points)
            min_required_linearity = 0.40  # For large clusters (>25 points)

            size_bonus = (cluster_size - min_cluster_size) / 2.0
            required_linearity = max(
                min_required_linearity,
                max_required_linearity - size_bonus * 0.035
            )

            if linearity_ratio < required_linearity:
                filtered_linearity += 1
                print(f"[Frontier] Rejected non-linear cluster: size={cluster_size}, "
                      f"linearity={linearity_ratio:.2f} (< {required_linearity:.2f} required for size {cluster_size}), "
                      f"endpoint_dist={endpoint_distance:.1f}px, path_length={path_length:.1f}px")
                continue

            # Get midpoint of this cluster
            mid_idx = len(sorted_points) // 2
            midpoint = sorted_points[mid_idx].astype(int)

            # Filter 3: Check if this midpoint is valid (not visited, far from agent)
            if not self._is_point_valid(midpoint, trajectory_exclusion_radius, agent_position, min_distance_from_agent):
                continue

            # Filter 4: Check distance to all existing valid midpoints
            too_close_to_existing = False
            for existing_midpoint in valid_midpoints:
                distance = np.linalg.norm(midpoint - existing_midpoint)
                if distance <= min_distance_between_midpoints:
                    too_close_to_existing = True
                    filtered_distance += 1
                    dist_meters = distance * self.map_resolution / 100.0
                    print(f"[Frontier] Rejected midpoint too close to existing: "
                          f"distance={dist_meters:.2f}m (< {min_distance_between_midpoints * self.map_resolution / 100.0:.2f}m)")
                    break

            if too_close_to_existing:
                continue

            # Filter 5: Hybrid connectivity check using distance + A* path planning
            # This is done last for performance (only check connectivity after passing all other filters)
            if distance_map is not None:
                row, col = int(midpoint[0]), int(midpoint[1])
                h, w = agent_region_mask.shape[0], agent_region_mask.shape[1]

                # Check bounds
                if not (0 <= row < h and 0 <= col < w):
                    filtered_connectivity += 1
                    print(f"[Frontier] Rejected frontier outside map bounds: midpoint={midpoint}")
                    continue

                # Get distance from midpoint to nearest point in agent's reachable region
                dist_to_agent_region = distance_map[row, col]

                # Hybrid strategy for better accuracy:
                # Stage 1: Quick accept/reject based on distance
                max_distance_threshold = 3  # Definitely connected
                max_distance_unreachable = 20  # Definitely too far

                if dist_to_agent_region <= max_distance_threshold:
                    # Close enough, definitely connected - accept directly
                    pass
                elif dist_to_agent_region > max_distance_unreachable:
                    # Too far away, definitely not reachable - reject directly
                    filtered_connectivity += 1
                    print(f"[Frontier] Rejected frontier too far from agent's region: "
                          f"midpoint={midpoint}, distance={dist_to_agent_region:.1f}px (> {max_distance_unreachable}px)")
                    continue
                else:
                    # Stage 2: Borderline case (3 < dist <= 20), use A* path planning for accurate check
                    # This handles cases where frontier is reachable through narrow corridors
                    if agent_position is not None and fbe_free_map is not None:
                        # Convert to numpy if needed
                        if torch.is_tensor(fbe_free_map):
                            fbe_free_np = fbe_free_map.cpu().numpy()
                        else:
                            fbe_free_np = fbe_free_map

                        # Squeeze extra dimensions if needed
                        if fbe_free_np.ndim > 2:
                            fbe_free_np = fbe_free_np.squeeze()
                            if fbe_free_np.ndim > 2:
                                fbe_free_np = fbe_free_np[..., 0]

                        # Use A* to check if path exists from agent to frontier
                        # Allow path length up to 2x the straight-line distance
                        max_allowed_path = dist_to_agent_region * 2
                        is_reachable = self._is_frontier_reachable(
                            midpoint, agent_position, fbe_free_np, max_path_length=max_allowed_path
                        )

                        if not is_reachable:
                            filtered_connectivity += 1
                            print(f"[Frontier] A* pathfinding rejected: midpoint={midpoint}, "
                                  f"distance={dist_to_agent_region:.1f}px (no valid path found)")
                            continue
                        else:
                            print(f"[Frontier] A* pathfinding accepted: midpoint={midpoint}, "
                                  f"distance={dist_to_agent_region:.1f}px (valid path found)")
                    else:
                        # No agent position or free map, fall back to distance threshold
                        filtered_connectivity += 1
                        print(f"[Frontier] Rejected frontier (no agent/map for pathfinding): "
                              f"midpoint={midpoint}, distance={dist_to_agent_region:.1f}px")
                        continue

            # Add to valid midpoints
            valid_midpoints.append(midpoint)

        # Print filtering statistics
        if filtered_size > 0:
            print(f"[Frontier] Filtered out {filtered_size} clusters (< {min_cluster_size} points)")
        if filtered_linearity > 0:
            print(f"[Frontier] Filtered out {filtered_linearity} clusters (linearity < {min_linearity_ratio})")
        if filtered_distance > 0:
            print(f"[Frontier] Filtered out {filtered_distance} midpoints (distance < {min_distance_between_midpoints * self.map_resolution / 100.0:.2f}m)")
        if filtered_connectivity > 0:
            print(f"[Frontier] Filtered out {filtered_connectivity} midpoints (connectivity: not reachable from agent's region)")

        if len(valid_midpoints) == 0:
            return None

        return np.array(valid_midpoints)

    def _is_point_valid(self, point, exclusion_radius, agent_position=None, min_distance_from_agent=15):
        """
        Check if a point is valid (unvisited area and far enough from agent).

        Args:
            point: [row, col] position in map coordinates
            exclusion_radius: Radius around point to check for visited areas
            agent_position: Tuple (agent_row, agent_col) in map coordinates. If None, no distance check is performed.
            min_distance_from_agent: Minimum distance from agent in pixels

        Returns:
            bool: True if the point is in an unvisited area and far enough from agent, False otherwise
        """
        # First check if point is in unvisited area
        if not self._is_point_unvisited(point, exclusion_radius):
            return False

        # Then check distance from agent if agent_position is provided
        if agent_position is not None:
            point_pos = np.array([800 - point[0], point[1]])
            agent_pos = np.array([agent_position[0], agent_position[1]])
            dist_to_agent = np.linalg.norm(point_pos - agent_pos)
            if dist_to_agent < min_distance_from_agent:
                return False

        return True

    def _is_point_unvisited(self, point, exclusion_radius):
        """
        Check if a point is in an unvisited area.

        Args:
            point: [row, col] position in map coordinates
            exclusion_radius: Radius around point to check

        Returns:
            bool: True if the area around the point is unvisited, False otherwise
        """
        if not hasattr(self.agent, 'agent_trajectory_map') or self.agent.agent_trajectory_map is None:
            # No trajectory map available, assume unvisited
            return True

        h, w = self.agent.agent_trajectory_map.shape
        point_r, point_c = int(self.map_height - point[0]), int(point[1])

        # Define the region around this point
        r_min = max(0, point_r - exclusion_radius)
        r_max = min(h, point_r + exclusion_radius + 1)
        c_min = max(0, point_c - exclusion_radius)
        c_max = min(w, point_c + exclusion_radius + 1)

        # Extract the region around the point
        region = self.agent.agent_trajectory_map[r_min:r_max, c_min:c_max]

        # Check if ALL cells in this region are unvisited (value == 0)
        if np.sum(region) == 0:
            return True  # All cells are 0 (unvisited)
        else:
            return False  # Some cells have been visited

    def _is_frontier_reachable(self, frontier_point, agent_position, fbe_free_map, max_path_length=None):
        """
        Check if a frontier is reachable from agent position using A* path planning.

        A* is a pathfinding algorithm that finds the shortest path from start to goal.
        It uses a heuristic (estimated distance to goal) to guide the search efficiently.

        Args:
            frontier_point: [row, col] of frontier midpoint
            agent_position: (row, col) of agent
            fbe_free_map: Free space map (1 = free, 0 = obstacle)
            max_path_length: Maximum allowed path length in pixels (None = no limit)

        Returns:
            bool: True if a valid path exists, False otherwise
        """
        import heapq

        # Convert positions to integers
        start = (int(agent_position[0]), int(agent_position[1]))
        goal = (int(frontier_point[0]), int(frontier_point[1]))

        # Quick check: if start or goal is out of bounds
        h, w = fbe_free_map.shape
        if not (0 <= start[0] < h and 0 <= start[1] < w):
            return False
        if not (0 <= goal[0] < h and 0 <= goal[1] < w):
            return False

        # Quick check: if goal is obstacle, not reachable
        if fbe_free_map[goal[0], goal[1]] != 1:
            return False

        # Quick check: if start is obstacle (shouldn't happen, but just in case)
        if fbe_free_map[start[0], start[1]] != 1:
            return False

        # A* search
        # Priority queue: stores (priority, position) tuples
        # Priority = cost_so_far + heuristic (estimated distance to goal)
        frontier_heap = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier_heap:
            _, current = heapq.heappop(frontier_heap)

            # Check if we reached the goal
            if current == goal:
                return True

            # Check all 8 neighbors (up, down, left, right, and diagonals)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # Skip current position

                    next_pos = (current[0] + dr, current[1] + dc)

                    # Check if next position is within map bounds
                    if not (0 <= next_pos[0] < h and 0 <= next_pos[1] < w):
                        continue

                    # Check if next position is free space
                    if fbe_free_map[next_pos[0], next_pos[1]] != 1:
                        continue

                    # Calculate cost to reach next position
                    # Diagonal moves cost 1.4 (sqrt(2)), straight moves cost 1.0
                    move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
                    new_cost = cost_so_far[current] + move_cost

                    # Early termination if path too long
                    if max_path_length is not None and new_cost > max_path_length:
                        continue

                    # If we found a better path to next_pos, update it
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        # Heuristic: Euclidean distance to goal
                        heuristic = np.sqrt((goal[0] - next_pos[0])**2 + (goal[1] - next_pos[1])**2)
                        priority = new_cost + heuristic
                        heapq.heappush(frontier_heap, (priority, next_pos))
                        came_from[next_pos] = current

        # No path found
        return False
