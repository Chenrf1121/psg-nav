"""
Landmark Geometry Analysis Module

Provides detailed geometric descriptions of landmarks to help LLM-based filtering
identify invalid landmarks (e.g., those in corners surrounded by explored areas).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2


class LandmarkGeometryAnalyzer:
    """Analyzes geometric properties of landmarks for validity checking."""

    def __init__(self, map_resolution: float = 5.0):
        """
        Initialize geometry analyzer.

        Args:
            map_resolution: Map resolution in cm/pixel
        """
        self.map_resolution = map_resolution

    def analyze_landmark(
        self,
        landmark_pos: np.ndarray,
        agent_pos: np.ndarray,
        fbe_free_map: np.ndarray,
        room_map: Optional[np.ndarray] = None,
        traversible: Optional[np.ndarray] = None,
        full_map: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Analyze geometric properties of a landmark.

        Args:
            landmark_pos: Landmark position [row, col]
            agent_pos: Agent position [row, col]
            fbe_free_map: Explored free space map (0-1 float)
            room_map: Room segmentation map (optional)
            traversible: Traversible map (optional)
            full_map: Full obstacle map (optional)

        Returns:
            Dictionary containing geometric descriptions
        """
        row, col = int(landmark_pos[0]), int(landmark_pos[1])
        agent_row, agent_col = int(agent_pos[0]), int(agent_pos[1])
        h, w = fbe_free_map.shape

        geometry = {}

        # ===== 1. Basic Position Information =====
        geometry['position'] = {
            'row': row,
            'col': col,
            'pixel_coordinates': f"({row}, {col})"
        }

        # Distance to agent
        dist_pixels = np.linalg.norm(landmark_pos - agent_pos)
        dist_meters = dist_pixels * self.map_resolution / 100.0
        geometry['distance_to_agent'] = {
            'pixels': float(dist_pixels),
            'meters': float(dist_meters),
            'description': f"{dist_meters:.1f}m away from agent"
        }

        # ===== 2. Exploration Status Around Landmark =====
        exploration_info = self._analyze_exploration_around_landmark(
            row, col, fbe_free_map, h, w
        )
        geometry['exploration'] = exploration_info

        # ===== 3. Spatial Configuration =====
        spatial_info = self._analyze_spatial_configuration(
            row, col, fbe_free_map, full_map, h, w
        )
        geometry['spatial_configuration'] = spatial_info

        # ===== 4. Directional Analysis =====
        directional_info = self._analyze_directions(
            row, col, fbe_free_map, h, w
        )
        geometry['directional_analysis'] = directional_info

        # ===== 5. Room Information =====
        if room_map is not None:
            room_info = self._analyze_room_context(row, col, room_map)
            geometry['room_context'] = room_info

        # ===== 6. Validity Assessment =====
        validity = self._assess_validity(geometry)
        geometry['validity_assessment'] = validity

        # ===== 7. Natural Language Description =====
        description = self._generate_description(geometry)
        geometry['natural_language_description'] = description

        return geometry

    def _analyze_exploration_around_landmark(
        self,
        row: int,
        col: int,
        fbe_free_map: np.ndarray,
        h: int,
        w: int
    ) -> Dict:
        """Analyze exploration status around the landmark."""
        exploration_info = {}

        # Check multiple radii
        radii = [5, 10, 20, 30]  # pixels
        for radius in radii:
            radius_meters = radius * self.map_resolution / 100.0

            # Define region
            r_min = max(0, row - radius)
            r_max = min(h, row + radius + 1)
            c_min = max(0, col - radius)
            c_max = min(w, col + radius + 1)

            region = fbe_free_map[r_min:r_max, c_min:c_max]

            # Calculate exploration ratio
            total_cells = region.size
            explored_cells = np.sum(region > 0.5)
            exploration_ratio = explored_cells / total_cells if total_cells > 0 else 0

            exploration_info[f'radius_{radius}px'] = {
                'radius_meters': float(radius_meters),
                'exploration_ratio': float(exploration_ratio),
                'explored_cells': int(explored_cells),
                'total_cells': int(total_cells)
            }

        # Overall assessment
        nearby_exploration = exploration_info['radius_10px']['exploration_ratio']
        if nearby_exploration > 0.9:
            status = "heavily_explored"
            status_desc = "surrounded by explored areas"
        elif nearby_exploration > 0.7:
            status = "mostly_explored"
            status_desc = "mostly explored around"
        elif nearby_exploration > 0.4:
            status = "partially_explored"
            status_desc = "partially explored around"
        else:
            status = "largely_unexplored"
            status_desc = "near unexplored areas"

        exploration_info['status'] = status
        exploration_info['status_description'] = status_desc

        return exploration_info

    def _analyze_spatial_configuration(
        self,
        row: int,
        col: int,
        fbe_free_map: np.ndarray,
        full_map: Optional[np.ndarray],
        h: int,
        w: int
    ) -> Dict:
        """Analyze spatial configuration (corner, corridor, open space, etc.)."""
        spatial_info = {}

        # Check if in corner (near two perpendicular boundaries)
        border_threshold = 20  # pixels from edge
        near_top = row < border_threshold
        near_bottom = row > h - border_threshold
        near_left = col < border_threshold
        near_right = col > w - border_threshold

        is_corner = (near_top or near_bottom) and (near_left or near_right)
        spatial_info['is_near_corner'] = is_corner

        if is_corner:
            corner_type = []
            if near_top: corner_type.append("top")
            if near_bottom: corner_type.append("bottom")
            if near_left: corner_type.append("left")
            if near_right: corner_type.append("right")
            spatial_info['corner_type'] = "-".join(corner_type)

        # Analyze openness (how much free space around)
        radius = 15
        r_min = max(0, row - radius)
        r_max = min(h, row + radius + 1)
        c_min = max(0, col - radius)
        c_max = min(w, col + radius + 1)

        region = fbe_free_map[r_min:r_max, c_min:c_max]
        openness = np.sum(region > 0.5) / region.size

        if openness > 0.8:
            space_type = "open_space"
            space_desc = "in an open area"
        elif openness > 0.5:
            space_type = "semi_open"
            space_desc = "in a semi-open area"
        elif openness > 0.3:
            space_type = "constrained"
            space_desc = "in a constrained space"
        else:
            space_type = "narrow"
            space_desc = "in a narrow space"

        spatial_info['space_type'] = space_type
        spatial_info['space_description'] = space_desc
        spatial_info['openness_ratio'] = float(openness)

        # Detect corridor (elongated free space in one direction)
        corridor_info = self._detect_corridor(row, col, fbe_free_map, h, w)
        spatial_info['corridor_detection'] = corridor_info

        # Obstacle density
        if full_map is not None:
            obstacle_region = full_map[r_min:r_max, c_min:c_max]
            obstacle_density = np.sum(obstacle_region > 0) / obstacle_region.size
            spatial_info['obstacle_density'] = float(obstacle_density)

        return spatial_info

    def _detect_corridor(
        self,
        row: int,
        col: int,
        fbe_free_map: np.ndarray,
        h: int,
        w: int
    ) -> Dict:
        """Detect if landmark is in a corridor."""
        corridor_info = {'is_corridor': False}

        # Sample in 4 directions
        sample_length = 20
        directions = {
            'north': (-1, 0),
            'south': (1, 0),
            'east': (0, 1),
            'west': (0, -1)
        }

        free_distances = {}
        for dir_name, (dr, dc) in directions.items():
            dist = 0
            for step in range(1, sample_length + 1):
                r = row + dr * step
                c = col + dc * step
                if 0 <= r < h and 0 <= c < w and fbe_free_map[r, c] > 0.5:
                    dist = step
                else:
                    break
            free_distances[dir_name] = dist

        # Check if elongated in one axis
        horizontal_extent = free_distances['east'] + free_distances['west']
        vertical_extent = free_distances['north'] + free_distances['south']

        if horizontal_extent > 15 and vertical_extent < 8:
            corridor_info['is_corridor'] = True
            corridor_info['orientation'] = 'horizontal'
        elif vertical_extent > 15 and horizontal_extent < 8:
            corridor_info['is_corridor'] = True
            corridor_info['orientation'] = 'vertical'

        corridor_info['free_distances'] = free_distances

        return corridor_info

    def _analyze_directions(
        self,
        row: int,
        col: int,
        fbe_free_map: np.ndarray,
        h: int,
        w: int
    ) -> Dict:
        """Analyze exploration status in different directions."""
        directions = {
            'north': (-1, 0),
            'northeast': (-1, 1),
            'east': (0, 1),
            'southeast': (1, 1),
            'south': (1, 0),
            'southwest': (1, -1),
            'west': (0, -1),
            'northwest': (-1, -1)
        }

        directional_info = {}
        sample_distance = 15  # pixels

        for dir_name, (dr, dc) in directions.items():
            # Sample along this direction
            explored_count = 0
            total_count = 0

            for step in range(1, sample_distance + 1):
                r = row + dr * step
                c = col + dc * step
                if 0 <= r < h and 0 <= c < w:
                    total_count += 1
                    if fbe_free_map[r, c] > 0.5:
                        explored_count += 1
                else:
                    break

            if total_count > 0:
                exploration_ratio = explored_count / total_count
                directional_info[dir_name] = {
                    'exploration_ratio': float(exploration_ratio),
                    'explored_cells': explored_count,
                    'total_cells': total_count,
                    'has_unexplored': exploration_ratio < 0.8
                }

        # Count directions with unexplored areas
        unexplored_directions = [
            dir_name for dir_name, info in directional_info.items()
            if info.get('has_unexplored', False)
        ]

        directional_info['summary'] = {
            'unexplored_direction_count': len(unexplored_directions),
            'unexplored_directions': unexplored_directions
        }

        return directional_info

    def _analyze_room_context(
        self,
        row: int,
        col: int,
        room_map: np.ndarray
    ) -> Dict:
        """Analyze room context of the landmark."""
        room_info = {}

        h, w = room_map.shape
        if 0 <= row < h and 0 <= col < w:
            room_id = int(room_map[row, col])
            room_info['room_id'] = room_id

            # Find room boundaries
            room_mask = (room_map == room_id)
            room_coords = np.argwhere(room_mask)

            if len(room_coords) > 0:
                room_center = room_coords.mean(axis=0)
                room_min = room_coords.min(axis=0)
                room_max = room_coords.max(axis=0)

                # Distance to room center
                dist_to_center = np.linalg.norm(np.array([row, col]) - room_center)
                room_size = np.linalg.norm(room_max - room_min)

                # Relative position in room
                if room_size > 0:
                    relative_dist = dist_to_center / room_size
                    if relative_dist < 0.3:
                        position_in_room = "center"
                    elif relative_dist < 0.6:
                        position_in_room = "mid"
                    else:
                        position_in_room = "edge"
                else:
                    position_in_room = "unknown"

                room_info['position_in_room'] = position_in_room
                room_info['distance_to_room_center'] = float(dist_to_center)
                room_info['room_size'] = float(room_size)

        return room_info

    def _assess_validity(self, geometry: Dict) -> Dict:
        """Assess landmark validity based on geometric properties."""
        validity = {
            'is_valid': True,
            'issues': [],
            'score': 1.0
        }

        exploration = geometry.get('exploration', {})
        spatial = geometry.get('spatial_configuration', {})
        directional = geometry.get('directional_analysis', {})

        # Check 1: Surrounded by explored areas (invalid)
        nearby_exploration = exploration.get('radius_10px', {}).get('exploration_ratio', 0)
        if nearby_exploration > 0.95:
            validity['issues'].append("Surrounded by heavily explored areas")
            validity['score'] *= 0.3

        # Check 2: In corner with no unexplored directions (invalid)
        is_corner = spatial.get('is_near_corner', False)
        unexplored_dir_count = directional.get('summary', {}).get('unexplored_direction_count', 0)

        if is_corner and unexplored_dir_count < 2:
            validity['issues'].append("In corner with few unexplored directions")
            validity['score'] *= 0.4

        # Check 3: Very high nearby exploration and low unexplored directions
        if nearby_exploration > 0.85 and unexplored_dir_count < 3:
            validity['issues'].append("High local exploration with limited exploration potential")
            validity['score'] *= 0.5

        # Check 4: Positive factors
        if unexplored_dir_count >= 5:
            validity['score'] *= 1.2  # Bonus for many unexplored directions

        # Final validity decision
        validity['is_valid'] = validity['score'] >= 0.5 and len(validity['issues']) < 2

        return validity

    def _generate_description(self, geometry: Dict) -> str:
        """Generate natural language description of the landmark."""
        parts = []

        # Distance
        dist_info = geometry.get('distance_to_agent', {})
        parts.append(f"Located {dist_info.get('meters', 0):.1f}m from agent")

        # Exploration status
        exploration = geometry.get('exploration', {})
        status_desc = exploration.get('status_description', '')
        if status_desc:
            parts.append(status_desc)

        # Spatial configuration
        spatial = geometry.get('spatial_configuration', {})
        space_desc = spatial.get('space_description', '')
        if space_desc:
            parts.append(space_desc)

        # Corner detection
        if spatial.get('is_near_corner', False):
            corner_type = spatial.get('corner_type', 'corner')
            parts.append(f"near {corner_type} corner")

        # Corridor detection
        corridor = spatial.get('corridor_detection', {})
        if corridor.get('is_corridor', False):
            orientation = corridor.get('orientation', '')
            parts.append(f"in a {orientation} corridor")

        # Unexplored directions
        directional = geometry.get('directional_analysis', {})
        unexplored_dirs = directional.get('summary', {}).get('unexplored_directions', [])
        if len(unexplored_dirs) > 0:
            parts.append(f"with unexplored areas toward {', '.join(unexplored_dirs[:3])}")
        else:
            parts.append("with no unexplored areas nearby")

        # Room context
        room = geometry.get('room_context', {})
        if room:
            room_pos = room.get('position_in_room', '')
            if room_pos:
                parts.append(f"at {room_pos} of room")

        # Validity
        validity = geometry.get('validity_assessment', {})
        if not validity.get('is_valid', True):
            issues = validity.get('issues', [])
            if issues:
                parts.append(f"⚠️ Issues: {'; '.join(issues)}")

        return ". ".join(parts) + "."


def analyze_all_landmarks(
    landmarks: np.ndarray,
    agent_pos: np.ndarray,
    fbe_free_map: np.ndarray,
    room_map: Optional[np.ndarray] = None,
    traversible: Optional[np.ndarray] = None,
    full_map: Optional[np.ndarray] = None,
    map_resolution: float = 5.0
) -> List[Dict]:
    """
    Analyze geometric properties for all landmarks.

    Args:
        landmarks: Nx2 array of landmark positions
        agent_pos: Agent position [row, col]
        fbe_free_map: Explored free space map
        room_map: Room segmentation map (optional)
        traversible: Traversible map (optional)
        full_map: Full obstacle map (optional)
        map_resolution: Map resolution in cm/pixel

    Returns:
        List of geometry dictionaries for each landmark
    """
    analyzer = LandmarkGeometryAnalyzer(map_resolution)

    results = []
    for i, landmark in enumerate(landmarks):
        geometry = analyzer.analyze_landmark(
            landmark, agent_pos, fbe_free_map,
            room_map, traversible, full_map
        )
        geometry['landmark_index'] = i
        results.append(geometry)

    return results


def filter_invalid_landmarks(
    landmarks: np.ndarray,
    geometries: List[Dict],
    min_validity_score: float = 0.5
) -> Tuple[np.ndarray, List[int]]:
    """
    Filter out invalid landmarks based on geometric analysis.

    Args:
        landmarks: Nx2 array of landmark positions
        geometries: List of geometry dictionaries from analyze_all_landmarks
        min_validity_score: Minimum validity score to keep landmark

    Returns:
        Tuple of (filtered_landmarks, valid_indices)
    """
    valid_indices = []

    for i, geometry in enumerate(geometries):
        validity = geometry.get('validity_assessment', {})
        score = validity.get('score', 0)

        if score >= min_validity_score:
            valid_indices.append(i)

    if len(valid_indices) > 0:
        filtered_landmarks = landmarks[valid_indices]
    else:
        filtered_landmarks = np.zeros((0, 2), dtype=float)

    return filtered_landmarks, valid_indices


def print_landmark_geometry_summary(geometries: List[Dict]):
    """Print a summary of landmark geometries for debugging."""
    print(f"\n{'='*80}")
    print(f"Landmark Geometry Analysis Summary ({len(geometries)} landmarks)")
    print(f"{'='*80}")

    for i, geo in enumerate(geometries):
        validity = geo.get('validity_assessment', {})
        is_valid = validity.get('is_valid', True)
        score = validity.get('score', 1.0)

        status_icon = "✓" if is_valid else "✗"

        print(f"\n{status_icon} Landmark {i} (validity score: {score:.2f})")
        print(f"  {geo.get('natural_language_description', 'No description')}")

        if not is_valid:
            issues = validity.get('issues', [])
            for issue in issues:
                print(f"    - {issue}")

    # Summary statistics
    valid_count = sum(1 for geo in geometries if geo.get('validity_assessment', {}).get('is_valid', True))
    invalid_count = len(geometries) - valid_count

    print(f"\n{'='*80}")
    print(f"Valid landmarks: {valid_count}/{len(geometries)} ({valid_count/len(geometries)*100:.1f}%)")
    print(f"Invalid landmarks: {invalid_count}/{len(geometries)} ({invalid_count/len(geometries)*100:.1f}%)")
    print(f"{'='*80}\n")
