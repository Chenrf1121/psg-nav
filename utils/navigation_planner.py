"""
Navigation Planner - Handles path planning and goal-related operations.

This module extracts navigation planning, topological path planning,
and goal-setting functionality from the main PSG_Nav_Agent class.
"""

import numpy as np
import skimage
from utils.utils_fmm.fmm_planner import FMMPlanner
import utils.utils_fmm.pose_utils as pu


class NavigationPlanner:
    """Manages navigation planning including topological paths and goal setting."""

    def __init__(self, agent):
        """
        Initialize NavigationPlanner.

        Args:
            agent: Reference to the main PSG_Nav_Agent instance
        """
        self.agent = agent
        self.map_resolution = agent.map_resolution
        self.map_size_cm = agent.map_size_cm

    def get_goal_gps(self, observations, angle, distance):
        """
        Calculate GPS coordinates for a goal at given angle and distance.

        Args:
            observations: Current observations including GPS and compass
            angle: Angle in degrees (can be numpy array or torch tensor)
            distance: Distance in meters

        Returns:
            numpy.ndarray: Goal GPS coordinates
        """
        import torch

        if type(angle) is torch.Tensor:
            angle = angle.cpu().numpy()
        agent_gps = observations['gps']
        agent_compass = observations['compass']
        goal_direction = agent_compass - angle/180*np.pi
        goal_gps = np.array([(agent_gps[0]+np.cos(goal_direction)*distance).item(),
         (agent_gps[1]-np.sin(goal_direction)*distance).item()])
        return goal_gps

    def get_relative_goal_gps(self, observations, goal_gps=None):
        """
        Calculate relative GPS distance to goal.

        Args:
            observations: Current observations
            goal_gps: Goal GPS coordinates (if None, uses self.agent.goal_gps)

        Returns:
            numpy.ndarray: [rho, phi] - distance and angle to goal
        """
        if goal_gps is None:
            goal_gps = self.agent.goal_gps
        direction_vector = goal_gps - np.array([observations['gps'][0].item(),observations['gps'][1].item()])
        rho = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
        phi_world = np.arctan2(direction_vector[1], direction_vector[0])
        agent_compass = observations['compass']
        phi = phi_world - agent_compass
        return np.array([rho, phi.item()], dtype=np.float32)

    def _check_direction_openness(self, agent_r, agent_c, direction, free_map, max_distance=15):
        """
        Check openness in a given direction (left or right).

        Args:
            agent_r: Agent row position
            agent_c: Agent column position
            direction: 1 for right, -1 for left
            free_map: Free space map tensor
            max_distance: Maximum distance to check (pixels)

        Returns:
            float: Openness score (higher = more open)
        """
        if free_map.dim() == 4:
            free_map_np = free_map.cpu().numpy()[0, 0]
        else:
            free_map_np = free_map.cpu().numpy()

        openness_scores = []
        sample_distances = [3, 6, 9, 12, 15]  # Sample at multiple distances

        for dist in sample_distances:
            if dist > max_distance:
                break

            # Sample points in forward-diagonal direction
            sample_r = int(agent_r - dist * 0.7)  # Forward component (70%)
            sample_c = int(agent_c + dist * 0.7 * direction)  # Side component (70%)

            # Check if within bounds
            if 0 <= sample_r < free_map_np.shape[0] and 0 <= sample_c < free_map_np.shape[1]:
                # Get free space value at this point
                openness_scores.append(free_map_np[sample_r, sample_c])
            else:
                # Out of bounds, penalize
                openness_scores.append(0)

        # Return average openness
        return np.mean(openness_scores) if len(openness_scores) > 0 else 0.0

    def find_most_open_wide_direction(self, agent_position, agent_orientation, free_map, scan_distance=30, angular_width=30):
        """
        Find the most open direction where not only the direction itself is open,
        but also the surrounding angles (±angular_width) are all open.

        Args:
            agent_position: (row, col) current agent position
            agent_orientation: Agent's current orientation in degrees (-180, 180)
            free_map: Free space map tensor
            scan_distance: How far to check in each direction (pixels)
            angular_width: How many degrees around the direction should also be open (default: 30)

        Returns:
            tuple: (best_angle_deg, goal_position, openness_score) or (None, None, 0) if no good direction
        """
        if free_map.dim() == 4:
            free_map_np = free_map.cpu().numpy()[0, 0]
        else:
            free_map_np = free_map.cpu().numpy()

        agent_r, agent_c = agent_position

        # Scan angles from -90 to +90 degrees (relative to agent's forward direction)
        # Negative = left, Positive = right, 0 = forward
        # Use progressive search: try narrow range first, expand if needed

        best_angle = None
        best_score = -1
        best_goal_pos = None

        # Progressive search ranges: prefer directions closer to forward
        search_ranges = [
            (-30, 30, "±30° (narrow - prefer forward)"),
            (-45, 45, "±45° (medium)"),
            (-60, 60, "±60° (wide - last resort)")
        ]

        for range_min, range_max, range_desc in search_ranges:
            relative_angles = np.arange(range_min, range_max + 1, 15)
            found_valid = False

            for relative_angle in relative_angles:
                # For each center angle, check if the angular_width around it is also open
                total_openness = 0
                samples_count = 0
                is_valid = True
                min_openness = 1.0

                # Check the center angle and surrounding angles (±angular_width)
                check_angles = np.arange(relative_angle - angular_width,
                                         relative_angle + angular_width + 1, 5)

                for angle_offset in check_angles:
                    # Convert relative angle to absolute map angle
                    # agent_orientation is in degrees (0-360), where:
                    #   0° = facing right (+col direction)
                    #   90° = facing up (-row direction)
                    #   180° = facing left (-col direction)
                    #   270° = facing down (+row direction)
                    absolute_angle = agent_orientation + angle_offset
                    angle_rad = np.deg2rad(absolute_angle)

                    # Sample points along this direction
                    # Only check within clearance distance (0.5m = 10 pixels at 5cm/pixel resolution)
                    min_clearance_pixels = 10  # 0.5 meters at 5cm/pixel resolution
                    check_distances = [d for d in [5, 10] if d <= min(scan_distance, min_clearance_pixels)]

                    for dist in check_distances:

                        # Calculate sample position in map coordinates
                        # cos(angle) gives col direction, -sin(angle) gives row direction
                        sample_r = int(agent_r - dist * np.sin(angle_rad))
                        sample_c = int(agent_c + dist * np.cos(angle_rad))

                        # Check bounds
                        if 0 <= sample_r < free_map_np.shape[0] and 0 <= sample_c < free_map_np.shape[1]:
                            openness = free_map_np[sample_r, sample_c]
                            min_openness = min(min_openness, openness)
                            # Very lenient threshold: just need 0.5m clearance, allow near-obstacles
                            # If openness > 0.02, consider it passable (not a solid wall)
                            if openness < 0.02:  # Only reject solid obstacles
                                is_valid = False
                                break
                            total_openness += openness
                            samples_count += 1
                        else:
                            is_valid = False
                            break

                    if not is_valid:
                        break

                if samples_count > 0:
                    avg_openness = total_openness / samples_count

                    if is_valid and avg_openness > best_score:
                        best_score = avg_openness
                        best_angle = relative_angle
                        found_valid = True

                        # Set goal position at scan_distance in the best direction
                        absolute_angle = agent_orientation + relative_angle
                        angle_rad = np.deg2rad(absolute_angle)
                        goal_r = int(agent_r - scan_distance * np.sin(angle_rad))
                        goal_c = int(agent_c + scan_distance * np.cos(angle_rad))

                        # Clip to bounds
                        goal_r = np.clip(goal_r, 0, free_map_np.shape[0] - 1)
                        goal_c = np.clip(goal_c, 0, free_map_np.shape[1] - 1)
                        best_goal_pos = (goal_r, goal_c)

            if found_valid:
                break

        if best_angle is not None:
            return best_angle, best_goal_pos, best_score
        else:
            return None, None, 0

    def find_most_open_direction_from_free_map(self, agent_position, previous_position=None,
                                                 move_distance=2, num_directions=16):
        """
        Find the most open direction using fbe_free_map and move in that direction.

        This method is used when the agent is stuck. It finds the direction with the most
        free space and returns a goal position in that direction.

        Args:
            agent_position: (row, col) current agent position in map coordinates
            previous_position: (row, col) previous agent position to avoid (optional)
            move_distance: How many units to move in the chosen direction (default: 2)
            num_directions: Number of directions to sample (default: 16)

        Returns:
            tuple: (goal_row, goal_col) representing the target position, or None if no valid direction
        """
        if not hasattr(self.agent, 'fbe_free_map'):
            return None

        free_map = self.agent.fbe_free_map.cpu().numpy()[0, 0]  # [H, W]
        agent_r, agent_c = agent_position

        # Sample directions around the agent
        angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
        best_direction = None
        best_openness_score = -1

        for angle in angles:
            # Calculate unit direction vector
            dir_r = np.cos(angle)
            dir_c = np.sin(angle)

            # Check if this direction points towards previous position (if provided)
            if previous_position is not None:
                prev_r, prev_c = previous_position
                # Vector from agent to previous position
                to_prev_r = prev_r - agent_r
                to_prev_c = prev_c - agent_c
                prev_dist = np.sqrt(to_prev_r**2 + to_prev_c**2)

                if prev_dist > 0.1:  # Avoid division by zero
                    to_prev_r /= prev_dist
                    to_prev_c /= prev_dist

                    # Dot product to check if directions are similar
                    dot_product = dir_r * to_prev_r + dir_c * to_prev_c

                    # Skip directions that point back to previous position (dot > 0.7 means < 45 degrees)
                    if dot_product > 0.7:
                        continue

            # Sample points along this direction and calculate openness score
            openness_score = 0
            sample_distances = [1, 2, 3, 4, 5]  # Sample at multiple distances

            for dist in sample_distances:
                sample_r = int(agent_r + dist * dir_r)
                sample_c = int(agent_c + dist * dir_c)

                # Check bounds
                if 0 <= sample_r < free_map.shape[0] and 0 <= sample_c < free_map.shape[1]:
                    # Add the free space value at this point
                    openness_score += free_map[sample_r, sample_c]
                else:
                    # Out of bounds, penalize this direction
                    openness_score -= 10
                    break

            # Update best direction if this one is more open
            if openness_score > best_openness_score:
                best_openness_score = openness_score
                best_direction = angle

        if best_direction is not None:
            # Calculate goal position in the most open direction
            goal_r = int(agent_r + move_distance * np.cos(best_direction))
            goal_c = int(agent_c + move_distance * np.sin(best_direction))

            # Clip to map bounds
            goal_r = np.clip(goal_r, 0, free_map.shape[0] - 1)
            goal_c = np.clip(goal_c, 0, free_map.shape[1] - 1)

            direction_deg = np.degrees(best_direction)
            print(f"[Unstuck] Moving towards most open direction: {direction_deg:.0f}°, openness score: {best_openness_score:.1f}")
            print(f"[Unstuck] Target position: ({goal_r}, {goal_c}), distance: {move_distance} units")

            return (goal_r, goal_c)
        else:
            print("[Unstuck] No valid open direction found")
            return None

    def _raycast_to_obstacle(self, agent_r, agent_c, angle_deg, free_map, max_distance=50):
        """
        Raycast from agent position in given direction until hitting obstacle.

        Args:
            agent_r, agent_c: Agent position
            angle_deg: Direction in degrees (map coordinates with flip)
            free_map: Free space map (already flipped with [::-1])
            max_distance: Maximum raycast distance in pixels

        Returns:
            int: Distance to nearest obstacle in pixels (or max_distance if no obstacle found)

        Coordinate system (with [::-1] flip):
            0° = right (+col), -90° = down (+row), 90° = up (-row), 180° = left (-col)
        """
        angle_rad = np.deg2rad(angle_deg)

        for dist in range(1, max_distance + 1):
            # Calculate position at this distance
            # Negate sin because map is flipped vertically
            check_r = int(agent_r - dist * np.sin(angle_rad))
            check_c = int(agent_c + dist * np.cos(angle_rad))

            # Check bounds (out of bounds = obstacle)
            if not (0 <= check_r < free_map.shape[0] and 0 <= check_c < free_map.shape[1]):
                return dist

            # Check if obstacle (free_map < 0.5 means obstacle)
            if free_map[check_r, check_c] < 0.5:
                return dist

        return max_distance

    def _visualize_stuck_recovery_scan(self, free_map, agent_r, agent_c, agent_orientation):
        """
        Visualize free_map with coordinate grid and scan directions for debugging.

        Args:
            free_map: Free space map (2D numpy array)
            agent_r, agent_c: Agent position in map coordinates
            agent_orientation: Agent orientation in degrees (-180 to 180)
        """
        import cv2
        import os

        # Create RGB visualization
        H, W = free_map.shape
        vis = np.zeros((H, W, 3), dtype=np.uint8)

        # Draw free_map (white=free, black=obstacle, gray=unknown)
        for r in range(H):
            for c in range(W):
                if free_map[r, c] >= 0.5:
                    vis[r, c] = [255, 255, 255]  # White - free space
                elif free_map[r, c] < 0.1:
                    vis[r, c] = [0, 0, 0]  # Black - obstacle
                else:
                    vis[r, c] = [128, 128, 128]  # Gray - unknown

        # Draw coordinate grid (every 50 pixels)
        grid_spacing = 50
        for x in range(0, W, grid_spacing):
            cv2.line(vis, (x, 0), (x, H), (100, 100, 255), 1)  # Light blue vertical lines
            if x % 100 == 0:
                cv2.putText(vis, str(x), (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        for y in range(0, H, grid_spacing):
            cv2.line(vis, (0, y), (W, y), (100, 100, 255), 1)  # Light blue horizontal lines
            if y % 100 == 0:
                cv2.putText(vis, str(y), (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

        # Draw agent position (large red circle)
        cv2.circle(vis, (agent_c, agent_r), 8, (0, 0, 255), -1)  # Red filled circle
        cv2.circle(vis, (agent_c, agent_r), 12, (0, 0, 255), 2)  # Red ring

        # Draw scan directions (every 30 degrees)
        ray_length = 40
        for angle in range(-180, 181, 30):
            angle_rad = np.deg2rad(angle)
            # Negate sin because map is flipped vertically
            end_r = int(agent_r - ray_length * np.sin(angle_rad))
            end_c = int(agent_c + ray_length * np.cos(angle_rad))

            # Different colors for different angle ranges
            if abs(angle - agent_orientation) <= 15:
                color = (0, 255, 0)  # Green - current orientation
                thickness = 2
            elif -45 <= (angle - agent_orientation) <= 45 or -45 <= (angle - agent_orientation + 360) <= 45 or -45 <= (angle - agent_orientation - 360) <= 45:
                color = (0, 165, 255)  # Orange - excluded range
                thickness = 1
            else:
                color = (255, 255, 0)  # Cyan - scan range
                thickness = 1

            cv2.line(vis, (agent_c, agent_r), (end_c, end_r), color, thickness)

            # Add angle labels
            # Negate sin because map is flipped vertically
            label_r = int(agent_r - (ray_length + 15) * np.sin(angle_rad))
            label_c = int(agent_c + (ray_length + 15) * np.cos(angle_rad))
            cv2.putText(vis, f"{angle}", (label_c - 10, label_r + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Add legend
        cv2.putText(vis, f"Agent Orientation: {agent_orientation:.1f} deg", (10, H - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, f"Agent Pos: ({agent_r}, {agent_c})", (10, H - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, "Coords: 0deg=Right, -90deg=Down, 90deg=Up, 180deg=Left", (10, H - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, "Green=Current, Orange=Excluded, Cyan=Scan", (10, H - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, "White=Free, Black=Obstacle, Gray=Unknown", (10, H - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save visualization
        save_dir = os.path.join(self.agent.visualization_dir, 'stuck_recovery_debug')
        os.makedirs(save_dir, exist_ok=True)

        timestep = getattr(self.agent, 'timestep', getattr(self.agent, 'total_steps', 0))
        filename = f'stuck_scan_step_{timestep:04d}.png'
        save_path = os.path.join(save_dir, filename)

        cv2.imwrite(save_path, vis)
        print(f"[DEBUG] Saved stuck recovery scan visualization: {save_path}")

    def find_open_direction_for_stuck_recovery(self, agent_map_pos, agent_orientation):
        """
        Find the most open direction for stuck recovery by scanning 150° range.

        Logic:
        1. Scan from -75° to +75° (150° total range) relative to stuck orientation (6 frames ago)
        2. EXCLUDE -45° to +45° range (too close to stuck direction, may be perception error)
        3. For each direction (sampled every 30°, matching agent's turn angle), calculate openness score
        4. Openness score = percentage of free space in ±20° sectors at 0.5m and 1.0m
        5. ALWAYS select direction with highest openness score (no minimum threshold)
        6. Return the absolute angle (in map coordinates, -180 to 180)

        Coordinate system (with [::-1] flip for better visualization):
        - 0° = right (+col)
        - -90° = down (+row)
        - 90° = up (-row)
        - ±180° = left (-col)

        Args:
            agent_map_pos: (row, col) agent position in map coordinates
            agent_orientation: Orientation from 6 frames ago, in range (-180, 180)

        Returns:
            tuple: (goal_map, goal_loc, absolute_angle) - absolute_angle is the target orientation in degrees
        """
        # Get maps
        if not hasattr(self.agent, 'fbe_free_map') or self.agent.fbe_free_map is None:
            return None, None, None

        free_map = self.agent.fbe_free_map.cpu().numpy()[0, 0, ::-1]  # Flip to match visualization

        agent_r, agent_c = agent_map_pos

        # Scan parameters
        angle_range = 180  # Total range: 150° (-75° to +75°)
        angle_step = 30  # Sample every 30° (matching agent's turn angle)
        excluded_range = 45  # Exclude ±45° around stuck direction
        check_distances = [5, 10]  # 0.5m and 1.0m at 5cm/pixel resolution
        sector_half_angle = 30  # ±30° sector
        goal_distance = 15  # 0.75m at 5cm/pixel resolution

        # Calculate openness score for each direction
        best_score = -1
        best_obstacle_dist = -1
        best_relative_angle = None
        angle_scores = []  # List of (relative_angle, openness_score, obstacle_distance, absolute_angle)

        for relative_angle in range(-90, 91, angle_step):  # -75, -45, -15, 15, 45, 75 (every 30°)
            # EXCLUDE -45° to +45° range (too close to stuck direction)
            if -excluded_range <= relative_angle <= excluded_range:
                continue

            # Calculate absolute angle in map coordinate system
            absolute_angle = agent_orientation + relative_angle

            # Normalize angle to (-180, 180) range
            while absolute_angle > 180:
                absolute_angle -= 360
            while absolute_angle < -180:
                absolute_angle += 360

            # Calculate distance to nearest obstacle in this direction
            obstacle_distance = self._raycast_to_obstacle(agent_r, agent_c, absolute_angle, free_map, max_distance=50)

            # Calculate openness score for this direction
            total_samples = 0
            free_samples = 0

            for check_dist in check_distances:
                # Check sector around this direction
                for sector_offset in range(-sector_half_angle, sector_half_angle + 1, 5):
                    # Angle for this sample point
                    sample_absolute_angle = absolute_angle + sector_offset
                    angle_rad = np.deg2rad(sample_absolute_angle)

                    # Calculate sample position
                    # Negate sin because map is flipped vertically
                    # Coordinate system: 0° = right (+col), -90° = down (+row), 90° = up (-row), 180° = left (-col)
                    sample_r = int(agent_r - check_dist * np.sin(angle_rad))
                    sample_c = int(agent_c + check_dist * np.cos(angle_rad))

                    # Check bounds
                    if not (0 <= sample_r < free_map.shape[0] and 0 <= sample_c < free_map.shape[1]):
                        total_samples += 1
                        # Out of bounds counts as obstacle
                        continue

                    # Check if position is free
                    total_samples += 1
                    if free_map[sample_r, sample_c] >= 0.5:
                        free_samples += 1

            # Calculate openness score (0.0 to 1.0)
            if total_samples > 0:
                openness_score = free_samples / total_samples
            else:
                openness_score = 0.0

            angle_scores.append((relative_angle, openness_score, obstacle_distance, absolute_angle))

            # Update best direction with three-tier logic:
            # 1. Prefer higher openness score
            # 2. If tied, prefer farther obstacle distance
            # 3. If still tied, prefer smaller absolute relative angle (closer to forward)
            should_update = False
            if openness_score > best_score:
                should_update = True
            elif openness_score == best_score:
                if obstacle_distance > best_obstacle_dist:
                    should_update = True
                elif obstacle_distance == best_obstacle_dist:
                    if best_relative_angle is None or abs(relative_angle) < abs(best_relative_angle):
                        should_update = True

            if should_update:
                best_score = openness_score
                best_obstacle_dist = obstacle_distance
                best_relative_angle = relative_angle

        # Check if found any valid direction (after exclusion)
        if best_relative_angle is None:
            return None, None, None

        # ALWAYS return the most open direction, no minimum threshold
        # Calculate goal position at 0.75m in the best direction
        absolute_angle = agent_orientation + best_relative_angle
        # Normalize angle to (-180, 180) range
        while absolute_angle > 180:
            absolute_angle -= 360
        while absolute_angle < -180:
            absolute_angle += 360

        goal_angle_rad = np.deg2rad(absolute_angle)
        # Negate sin because map is flipped vertically
        goal_r = int(agent_r - goal_distance * np.sin(goal_angle_rad))
        goal_c = int(agent_c + goal_distance * np.cos(goal_angle_rad))

        # Clip to map bounds
        goal_r = np.clip(goal_r, 0, free_map.shape[0] - 1)
        goal_c = np.clip(goal_c, 0, free_map.shape[1] - 1)

        # Create goal map
        goal_map = np.zeros((free_map.shape[0], free_map.shape[1]))
        goal_map[goal_r, goal_c] = 1

        # goal_loc is in original coordinate system (before flip)
        goal_loc = np.array([goal_r, goal_c])

        # Find obstacle distance for the selected direction
        selected_obstacle_dist = None
        for rel_angle, score, obs_dist, abs_ang in angle_scores:
            if rel_angle == best_relative_angle:
                selected_obstacle_dist = obs_dist
                break

        # Return absolute angle directly (not relative angle)
        return goal_map, goal_loc, absolute_angle
