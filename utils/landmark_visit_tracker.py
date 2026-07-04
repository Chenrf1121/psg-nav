"""
Landmark Visit Tracker: Track visited landmarks and explored regions

This module prevents the robot from revisiting the same landmarks,
addressing the "wandering in explored areas" limitation.

Core features:
- Track which landmarks have been visited (one-time visit only)
- Mark explored regions around visited landmarks
- Compute novelty scores for new landmarks
"""

import numpy as np
import cv2


class LandmarkVisitTracker:
    """
    Track landmark visits and explored regions to avoid redundant exploration.

    Key principle: Each landmark can only be visited ONCE.
    Once visited, the surrounding region is marked as "explored".

    Attributes:
        map_size (int): Size of the occupancy map (pixels)
        resolution (float): Map resolution in cm/pixel
        visit_radius (float): Radius around visited landmarks marked as explored (meters)
        visited_landmarks (set): Set of visited landmark indices
        explored_map (np.ndarray): Binary map of explored regions (0=unexplored, 1=explored)
        visit_count_map (np.ndarray): Count of visits per map cell
    """

    def __init__(self, map_size, resolution=5.0, visit_radius=1.5):
        """
        Initialize the visit tracker.

        Args:
            map_size (int): Size of the map in pixels
            resolution (float): Map resolution in cm/pixel (default: 5.0)
            visit_radius (float): Radius in meters around visited landmarks to mark as explored (default: 1.5m)
        """
        self.map_size = map_size
        self.resolution = resolution
        self.visit_radius = visit_radius

        # Convert visit radius from meters to pixels
        self.visit_radius_pixels = int(visit_radius * 100 / resolution)

        # Set of visited landmark indices (each landmark can only be visited once)
        self.visited_landmarks = set()

        # Binary map: 0 = unexplored, 1 = explored
        self.explored_map = np.zeros((map_size, map_size), dtype=np.uint8)

        # Visit count per cell (for debugging/visualization)
        self.visit_count_map = np.zeros((map_size, map_size), dtype=np.int32)

        # Statistics
        self.total_visits = 0
        self.total_explored_area = 0

    def mark_visit(self, position, landmark_idx):
        """
        Mark a landmark as visited and update explored regions.

        Args:
            position: [row, col] position of the landmark in map coordinates
            landmark_idx (int): Index of the landmark being visited

        Returns:
            bool: True if this is a new visit, False if already visited
        """
        # Check if already visited
        if landmark_idx in self.visited_landmarks:
            return False

        # Mark as visited
        self.visited_landmarks.add(landmark_idx)
        self.total_visits += 1

        # Mark explored region around this landmark
        row, col = int(position[0]), int(position[1])

        # Ensure within bounds
        row = np.clip(row, 0, self.map_size - 1)
        col = np.clip(col, 0, self.map_size - 1)

        # Create circular mask around the landmark
        y, x = np.ogrid[-row:self.map_size-row, -col:self.map_size-col]
        mask = x*x + y*y <= self.visit_radius_pixels*self.visit_radius_pixels

        # Update explored map
        self.explored_map[mask] = 1

        # Update visit count map
        self.visit_count_map[mask] += 1

        # Update statistics
        self.total_explored_area = np.sum(self.explored_map)

        return True

    def is_visited(self, landmark_idx):
        """
        Check if a landmark has been visited.

        Args:
            landmark_idx (int): Index of the landmark to check

        Returns:
            bool: True if visited, False otherwise
        """
        return landmark_idx in self.visited_landmarks

    def get_novelty_score(self, position, search_radius=None):
        """
        Compute novelty score for a position (0-1).

        Higher score = more unexplored area nearby = more novel

        Args:
            position: [row, col] position to evaluate
            search_radius (float): Optional search radius in pixels.
                                  If None, uses visit_radius_pixels

        Returns:
            float: Novelty score (0=fully explored, 1=fully unexplored)
        """
        if search_radius is None:
            search_radius = self.visit_radius_pixels

        row, col = int(position[0]), int(position[1])

        # Ensure within bounds
        row = np.clip(row, 0, self.map_size - 1)
        col = np.clip(col, 0, self.map_size - 1)

        # Define search region
        r_min = max(0, row - search_radius)
        r_max = min(self.map_size, row + search_radius + 1)
        c_min = max(0, col - search_radius)
        c_max = min(self.map_size, col + search_radius + 1)

        # Extract region
        region = self.explored_map[r_min:r_max, c_min:c_max]

        if region.size == 0:
            return 1.0  # Empty region is novel by default

        # Calculate explored ratio
        explored_ratio = np.sum(region) / region.size

        # Novelty = 1 - explored_ratio
        novelty = 1.0 - explored_ratio

        return novelty

    def get_visit_penalty(self, landmark_idx):
        """
        Get visit penalty for a landmark (0-1).

        Since each landmark can only be visited once, penalty is binary:
        - 0.0 if not visited (no penalty)
        - 1.0 if visited (full penalty, should be excluded)

        Args:
            landmark_idx (int): Index of the landmark

        Returns:
            float: Penalty score (0=not visited, 1=visited)
        """
        return 1.0 if self.is_visited(landmark_idx) else 0.0

    def is_explored(self, position, threshold=0.7, search_radius=None):
        """
        Check if a position is sufficiently explored.

        Args:
            position: [row, col] position to check
            threshold (float): Explored ratio threshold (default: 0.7)
            search_radius (int): Search radius in pixels (default: visit_radius_pixels)

        Returns:
            bool: True if explored ratio > threshold
        """
        novelty = self.get_novelty_score(position, search_radius)
        explored_ratio = 1.0 - novelty
        return explored_ratio > threshold

    def filter_unvisited_landmarks(self, landmark_positions, landmark_indices=None):
        """
        Filter out visited landmarks from a list.

        Args:
            landmark_positions: Nx2 array of landmark positions
            landmark_indices: Optional array of landmark indices.
                            If None, uses array indices as landmark IDs.

        Returns:
            tuple: (filtered_positions, filtered_indices)
        """
        if landmark_positions is None or len(landmark_positions) == 0:
            return np.zeros((0, 2)), np.array([])

        if landmark_indices is None:
            landmark_indices = np.arange(len(landmark_positions))

        # Filter out visited landmarks
        unvisited_mask = np.array([
            not self.is_visited(idx) for idx in landmark_indices
        ])

        filtered_positions = landmark_positions[unvisited_mask]
        filtered_indices = landmark_indices[unvisited_mask]

        return filtered_positions, filtered_indices

    def get_stats(self):
        """
        Get statistics about visited landmarks and explored areas.

        Returns:
            dict: Statistics including visit count, explored area, etc.
        """
        total_map_area = self.map_size * self.map_size
        explored_percentage = (self.total_explored_area / total_map_area) * 100

        return {
            'total_visits': self.total_visits,
            'visited_landmarks': len(self.visited_landmarks),
            'explored_area_pixels': self.total_explored_area,
            'explored_percentage': explored_percentage,
            'visit_radius_meters': self.visit_radius,
            'visit_radius_pixels': self.visit_radius_pixels,
        }

    def reset(self):
        """Reset all tracking data (for new episode)."""
        self.visited_landmarks.clear()
        self.explored_map.fill(0)
        self.visit_count_map.fill(0)
        self.total_visits = 0
        self.total_explored_area = 0

    def visualize_explored_map(self, occupancy_map=None, agent_position=None):
        """
        Create a visualization of the explored map.

        Args:
            occupancy_map: Optional background map to overlay
            agent_position: Optional [row, col] of agent for visualization

        Returns:
            np.ndarray: RGB visualization image
        """
        # Create base image
        if occupancy_map is not None:
            # Use occupancy map as background (grayscale)
            vis = (occupancy_map * 255).astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        else:
            vis = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        # Overlay explored regions in green
        explored_overlay = np.zeros_like(vis)
        explored_overlay[self.explored_map == 1] = [0, 255, 0]  # Green for explored
        vis = cv2.addWeighted(vis, 0.7, explored_overlay, 0.3, 0)

        # Mark agent position if provided
        if agent_position is not None:
            agent_row, agent_col = int(agent_position[0]), int(agent_position[1])
            cv2.circle(vis, (agent_col, agent_row), 5, (255, 0, 0), -1)  # Blue dot

        return vis

    def __repr__(self):
        stats = self.get_stats()
        return (f"LandmarkVisitTracker("
                f"visited={stats['visited_landmarks']}, "
                f"explored={stats['explored_percentage']:.1f}%)")
