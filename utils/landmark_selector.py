"""
Landmark Selector - Handles landmark-scenegraph pairing and LLM-based ranking.

This module extracts the landmark selection logic from the main PSG_Nav_Agent class,
including comparison-based LLM ranking and fallback selection.
"""

import numpy as np
import json
import cv2
from PIL import Image
from graph.meta_func.flask_func import FlaskFunc


class LandmarkSelector(FlaskFunc):
    """Handles landmark selection through pairing and LLM ranking."""

    def __init__(self, agent):
        """
        Initialize LandmarkSelector.

        Args:
            agent: Reference to the main PSG_Nav_Agent instance
        """
        # Initialize FlaskFunc for LLM access
        server_port = getattr(agent.args, 'server_port', None)
        super().__init__(server_port=server_port)

        self.agent = agent
        self.map_resolution = agent.map_resolution
        self.map_height = agent.map_height

    def select_best_landmark_by_comparison(
        self,
        landmark_nodes: np.ndarray,
        scenegraph_combinations,
        object_nodes,
        agent_map_pos: tuple,
        obj_goal_sg: str,
    ):
        """
        Select the best landmark using comparison-based algorithm with LLM reasoning.

        This method implements a pairwise comparison approach where:
        1. Each landmark is enriched with spatial semantic descriptions
        2. Invalid landmarks (e.g., in corners with no exploration potential) are filtered
        3. Compute environmental semantic entropy for each landmark
        4. LLM computes prior probability of finding goal for each landmark
        5. Landmarks are compared pairwise in different scene contexts
        6. Relative Expected Utility is computed to select the best landmark

        Args:
            landmark_nodes: Nx2 array of landmark positions [[row, col], ...]
            scenegraph_combinations: SceneGraphCombinations object
            object_nodes: List of detected object nodes
            agent_map_pos: Tuple (agent_row, agent_col) in map coordinates
            obj_goal_sg: Goal object description
            use_llm_filter: If True, use LLM for filtering; if False, use rule-based filtering (ignored if use_vlm_filter=True)
            use_vlm_filter: If True, use VLM with occupancy map for filtering (highest priority)

        Returns:
            Tuple (goal_position, landmark_idx, explanation)
            - goal_position: np.array [row, col] for navigation goal
            - landmark_idx: Index of selected landmark in original landmark_nodes
            - explanation: String explanation of the selection
        """

        # Step 1: Add spatial semantics to landmarks
        landmarks_with_semantics = self._add_spatial_semantics_to_landmarks(
            landmark_nodes, agent_map_pos
        )

        # Step 2: Filter invalid landmarks based on geometry
        valid_landmarks, filtered_count = self._filter_invalid_landmarks_by_geometry(
            landmarks_with_semantics, use_llm=False, use_vlm=False
        )

        if len(valid_landmarks) == 0:
            return None, None, "No valid landmarks available"

        # Step 2.5: Compute environmental semantic entropy for each landmark
        valid_landmarks = self._compute_environmental_semantic_entropy(
            valid_landmarks, object_nodes
        )


        valid_landmarks = self._compute_landmark_goal_prior_probabilities(
            valid_landmarks, obj_goal_sg, scenegraph_combinations
        )

        # Step 4: Perform pairwise comparisons
        if len(valid_landmarks) > 1:
            comparison_results = self._compare_landmarks_pairwise(
                valid_landmarks, scenegraph_combinations, obj_goal_sg
            )
        else:
            comparison_results = {
                'comparisons': [],
                'win_counts': {0: 0},
                'total_comparisons': 0
            }

        # Step 5: Compute Relative Expected Utility and select best
        best_landmark_idx, best_utility, utilities = self._compute_relative_expected_utility(
            valid_landmarks, comparison_results
        )

        # Map back to original landmark index
        original_idx = valid_landmarks[best_landmark_idx]['original_index']
        best_position = landmark_nodes[original_idx]

        # Generate explanation
        explanation = self._generate_comparison_explanation(
            valid_landmarks[best_landmark_idx], best_utility, comparison_results
        )

        goal_from_landmark = np.array(best_position) - 1


        return goal_from_landmark, original_idx, explanation

    def _add_spatial_semantics_to_landmarks(self, landmark_nodes, agent_map_pos):
        """
        Step 1: Add spatial semantic descriptions to each landmark.

        Uses geometry analysis to describe:
        - Distance to agent
        - Surrounding exploration status
        - Spatial configuration (corner, corridor, open space)
        - Directional analysis (which directions have unexplored areas)
        - Room context

        Args:
            landmark_nodes: Nx2 array of landmark positions
            agent_map_pos: Agent position tuple

        Returns:
            List of landmark dictionaries with semantic information
        """

        landmarks = []
        agent_y, agent_x = agent_map_pos

        for i, lm_pos in enumerate(landmark_nodes):
            # Flip landmark row coordinate to match map coordinates
            # landmark_nodes stores in [row, col] but needs to be flipped
            row = int(self.map_height - lm_pos[0])  # Flip row coordinate
            col = int(lm_pos[1])

            # 1. Distance to agent (meters)
            dist_pixels = np.linalg.norm(np.array([row, col]) - np.array([agent_y, agent_x]))
            dist_meters = dist_pixels * self.map_resolution / 100.0

            # 2. Check if near frontier (< 0.2 meters)
            # Use flipped coordinates for lookup
            lm_key = (int(lm_pos[0]), int(lm_pos[1]))  # Original coordinates for lookup
            near_frontier = False
            frontier_dist_meters = None
            if hasattr(self.agent, 'landmark_near_frontier') and lm_key in self.agent.landmark_near_frontier:
                frontier_info = self.agent.landmark_near_frontier[lm_key]
                frontier_dist_meters = frontier_info['distance_to_frontier_meters']
                # Stricter threshold: 0.2m = 20cm = 4 pixels (at 5cm/pixel)
                near_frontier = frontier_dist_meters < 0.2

            # 3. Compute exploration status around landmark (use flipped coordinates)
            exploration_status = self._compute_exploration_status(row, col)

            # 4. Room information - use ORIGINAL coordinates (not flipped)
            # room_map uses the same coordinate system as landmark_nodes
            room_name = "unknown"
            room_prob = 0.0
            nearest_room_info = None  # Store nearest room info for unknown landmarks

            if hasattr(self.agent, 'room_map') and self.agent.room_map is not None:
                try:
                    # Use original coordinates (no flip) - confirmed correct
                    row_room = int(lm_pos[0])
                    col_room = int(lm_pos[1])
                    room_probs = self.agent.room_map[0, :, row_room, col_room].cpu().numpy()
                    max_prob = room_probs.max()


                    # Lower threshold from 0.1 to 0.05 to catch more room predictions
                    if max_prob > 0.05:
                        room_idx = room_probs.argmax()
                        if room_idx < len(self.agent.rooms):
                            room_name = self.agent.rooms[room_idx]
                            room_prob = room_probs[room_idx]
                    else:
                        # Room is unknown - find nearest known room
                        from scipy.ndimage import distance_transform_edt

                        min_distance = float('inf')
                        nearest_room_name = None

                        # Check each room channel
                        for room_idx in range(self.agent.room_map.shape[1]):
                            # Get room probability map for this room
                            room_channel = self.agent.room_map[0, room_idx, :, :].cpu().numpy()

                            # Create binary mask of areas with high room probability (>0.1)
                            room_mask = (room_channel > 0.1).astype(np.uint8)

                            if room_mask.sum() == 0:
                                # No detected area for this room
                                continue

                            # Compute distance transform from this room's area
                            inverted_mask = 1 - room_mask
                            distance_map = distance_transform_edt(inverted_mask)

                            # Get distance from landmark to this room
                            dist_to_room = distance_map[row_room, col_room]

                            if dist_to_room < min_distance:
                                min_distance = dist_to_room
                                nearest_room_name = self.agent.rooms[room_idx]

                        if nearest_room_name is not None:
                            # Convert distance from pixels to meters
                            dist_to_nearest_room_meters = min_distance * self.map_resolution / 100.0
                            nearest_room_info = {
                                'room_name': nearest_room_name,
                                'distance_meters': dist_to_nearest_room_meters,
                                'distance_pixels': min_distance
                            }


                except Exception as e:
                    import traceback
                    traceback.print_exc()

            # 5. Generate natural language description
            semantics = self._generate_landmark_description(
                dist_meters, near_frontier, frontier_dist_meters,
                exploration_status, room_name, room_prob
            )

            # 6. Store landmark information
            landmarks.append({
                'original_index': i,
                'position': lm_pos,  # Store original position
                'position_flipped': np.array([row, col]),  # Store flipped position for map access
                'geometry': {
                    'distance_to_agent_meters': dist_meters,
                    'near_frontier': near_frontier,
                    'frontier_distance_meters': frontier_dist_meters,
                    'exploration_pct': exploration_status['exploration_pct'],
                    'explored_fully': exploration_status['explored_fully'],
                    'info_gain': exploration_status['info_gain'],  # Add info_gain
                    'max_possible_info': exploration_status['max_possible_info'],  # Add max_possible_info
                    'room_name': room_name,
                    'room_probability': room_prob,
                    'nearest_room_info': nearest_room_info  # Add nearest room info for unknown rooms
                },
                'semantics': semantics,
                'prior_probability': 0.0,
                'win_count': 0,
                'loss_count': 0,
                'total_comparisons': 0
            })
        return landmarks

    def _filter_invalid_landmarks_by_geometry(self, landmarks, use_llm=True, use_vlm=False):
        """
        Step 2: Filter out spatially invalid landmarks.

        Supports three methods:
        1. VLM-based: Uses VLM with occupancy map visualization (most intelligent)
        2. LLM-based: Uses LLM reasoning (slower but more intelligent)
        3. Rule-based: Uses simple geometric rules (fast)

        A landmark is considered invalid if:
        - It cannot expand the explored area (heavily explored with no frontiers)
        - It has no exploration potential
        - It is in a spatial dead-end with no exploration value
        - It is unreachable due to obstacles

        Note: This step only uses spatial/geometric information, not scene graph objects.

        Args:
            landmarks: List of landmarks with semantic information
            use_llm: If True, use LLM; if False, use rule-based filtering (ignored if use_vlm=True)
            use_vlm: If True, use VLM with occupancy map visualization (highest priority)

        Returns:
            Tuple (valid_landmarks, filtered_count)
        """
        if use_vlm:
            return self._filter_landmarks_by_vlm(landmarks)
        elif use_llm:
            return self._filter_landmarks_by_llm(landmarks)
        else:
            return self._filter_landmarks_by_rules(landmarks)

    def _create_occupancy_map_image(self, landmarks, agent_map_pos):
        """
        Create an occupancy map visualization with landmarks and agent position.
        Uses the same visualization style as the video generation.

        Args:
            landmarks: List of landmarks with position information
            agent_map_pos: Tuple (agent_row, agent_col) in map coordinates

        Returns:
            PIL Image: RGB image of the occupancy map with landmarks and agent marked
        """
        import torch
        import skimage.morphology
        from matplotlib import colors
        import cv2
        from scipy.ndimage import binary_dilation

        # Get map dimensions
        H, W = self.agent.full_map.shape[-2:]

        # Create base map similar to video generation (without needing traversible)
        # Start with white background (unknown areas)
        paper_map_trans = torch.ones((H, W, 3), dtype=torch.float32)  # White, use float32
        unknown_rgb = colors.to_rgb('#FFFFFF')
        paper_map_trans[:, :, :] = torch.tensor(unknown_rgb, dtype=torch.float32)

        # Draw free space (light gray)
        free_rgb = colors.to_rgb('#E7E7E7')
        free_mask = self.agent.fbe_free_map.cpu().numpy()[0, 0, ::-1] > 0.5
        paper_map_trans[free_mask, :] = torch.tensor(free_rgb, dtype=torch.float32)

        # Draw obstacles (darker gray) with dilation
        obstacle_rgb = colors.to_rgb('#A2A2A2')
        obs_mask = skimage.morphology.binary_dilation(
            self.agent.full_map.cpu().numpy()[0, 0, ::-1] > 0.5,
            skimage.morphology.disk(1)
        )
        paper_map_trans[obs_mask, :] = torch.tensor(obstacle_rgb, dtype=torch.float32)

        # Convert to channel-first format for visualization
        paper_map_trans_chw = paper_map_trans.permute(2, 0, 1)

        # Calculate and draw frontiers (yellow)
        from utils.utils_frontiers import calculate_frontiers
        frontier_map, frontier_locations, num_frontiers = calculate_frontiers(
            self.agent.full_map, self.agent.fbe_free_map
        )

        # Draw frontiers before agent (so agent is on top)
        if frontier_locations is not None and len(frontier_locations) > 0:
            frontier_color = (1.0, 1.0, 0.0)  # Yellow
            frontier_size = 1

            # Convert frontier locations to numpy if needed
            if torch.is_tensor(frontier_locations):
                fl_np = frontier_locations.cpu().numpy()
            else:
                fl_np = frontier_locations

            # Transform frontier coordinates
            for frontier in fl_np:
                try:
                    r = int(frontier[0])
                    c = int(frontier[1])
                except Exception:
                    continue

                # Use same coordinate transform as landmarks
                top = int(self.agent.map_size_cm/5) - r - frontier_size
                bottom = int(self.agent.map_size_cm/5) - r + frontier_size
                left = c - frontier_size
                right = c + frontier_size

                # Clip to bounds
                H_ch, W_ch = paper_map_trans_chw.shape[1], paper_map_trans_chw.shape[2]
                top = max(0, top)
                left = max(0, left)
                bottom = min(H_ch, bottom)
                right = min(W_ch, right)

                if top >= bottom or left >= right:
                    continue

                # Draw yellow frontier
                paper_map_trans_chw[0, top:bottom, left:right] = frontier_color[0]
                paper_map_trans_chw[1, top:bottom, left:right] = frontier_color[1]
                paper_map_trans_chw[2, top:bottom, left:right] = frontier_color[2]

        # Draw agent position only (no goal, no edges)
        from utils.image_process import draw_agent
        for idx, pose in enumerate(self.agent.history_pose):
            draw_step_num = 30
            alpha = max(0, 1 - (len(self.agent.history_pose) - idx) / draw_step_num)
            agent_size = 1
            if idx == len(self.agent.history_pose) - 1:
                agent_size = 2
            draw_agent(agent=self.agent, map=paper_map_trans_chw, pose=pose,
                      agent_size=agent_size, color_index=0, alpha=alpha)

        # Convert to RGB numpy array
        occupancy_map_full = (paper_map_trans_chw.permute(1, 2, 0) * 255).numpy().astype(np.uint8)

        # Overlay agent trajectory history (same as video generation)
        if hasattr(self.agent, 'agent_trajectory_map') and self.agent.agent_trajectory_map is not None:
            trajectory_map = self.agent.agent_trajectory_map
            H_traj, W_traj = trajectory_map.shape
            H_map, W_map = occupancy_map_full.shape[:2]

            # Expand visited areas
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
            expanded_trajectory = binary_dilation(trajectory_map > 0, structure=kernel).astype(np.uint8)

            # Light green for expanded areas
            light_green = np.array([150, 255, 150], dtype=np.uint8)
            alpha_light = 0.2

            # Medium green for core path
            medium_green = np.array([100, 220, 100], dtype=np.uint8)
            alpha_core = 0.35

            # Apply overlays
            mask_expanded = expanded_trajectory > 0
            if H_traj == H_map and W_traj == W_map:
                occupancy_map_full[mask_expanded] = (
                    occupancy_map_full[mask_expanded] * (1 - alpha_light) +
                    light_green * alpha_light
                ).astype(np.uint8)

                mask_core = trajectory_map > 0
                occupancy_map_full[mask_core] = (
                    occupancy_map_full[mask_core] * (1 - alpha_core) +
                    medium_green * alpha_core
                ).astype(np.uint8)

        # Draw candidate landmarks with numbers (same style as video but with numbers)
        # Use the same coordinate transform as video generation
        def transform_coords(r, c):
            y = int(int(self.agent.map_size_cm/5) - int(r))
            x = int(c)
            return (x, y)

        H_map, W_map = occupancy_map_full.shape[:2]
        landmark_size = 4  # Same size as in video

        for i, lm in enumerate(landmarks):
            # Use position (original [row, col] format)
            pos = lm['position']
            r, c = int(pos[0]), int(pos[1])

            lm_x, lm_y = transform_coords(r, c)

            # Draw landmark as blue square (same as video)
            top = lm_y - landmark_size
            bottom = lm_y + landmark_size
            left = lm_x - landmark_size
            right = lm_x + landmark_size

            # Clip to bounds
            top = max(0, top)
            left = max(0, left)
            bottom = min(H_map, bottom)
            right = min(W_map, right)

            if top >= bottom or left >= right:
                continue

            # Draw blue square (RGB format: 0,0,255)
            occupancy_map_full[top:bottom, left:right] = [0, 0, 255]

            # Draw landmark number next to it (0-indexed to match video)
            text = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1

            # Position text to the right of the landmark
            text_x = right + 2
            text_y = lm_y + 3

            # Ensure text is within bounds
            if text_x < W_map - 10 and 0 <= text_y < H_map:
                # Draw with white background for visibility
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                # Draw semi-transparent background
                cv2.rectangle(occupancy_map_full,
                            (text_x - 1, text_y - text_h - 1),
                            (text_x + text_w + 1, text_y + 2),
                            (255, 255, 255), -1)
                # Draw black text
                cv2.putText(occupancy_map_full, text, (text_x, text_y),
                          font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Convert to PIL Image
        image = Image.fromarray(occupancy_map_full)
        return image

    def _filter_landmarks_by_vlm(self, landmarks):
        """
        VLM-based filtering: Use VLM with occupancy map to judge landmark validity.

        This method evaluates each landmark individually while showing all landmarks:
        1. Creates a single occupancy map showing ALL landmarks
        2. For each landmark, asks VLM if removing it would affect navigation
        3. Collects results from all evaluations

        Args:
            landmarks: List of landmarks with semantic information

        Returns:
            Tuple (valid_landmarks, filtered_count)
        """

        if len(landmarks) == 0:
            return landmarks, 0

        # Get agent position
        if hasattr(self.agent, 'last_sim_location'):
            agent_x, agent_y = self.agent.last_sim_location[0], self.agent.last_sim_location[1]
            agent_map_pos = (
                int((self.agent.map_size_cm / 100 - agent_y) * 100 / self.agent.resolution),
                int(agent_x * 100 / self.agent.resolution)
            )
        else:
            agent_map_pos = (self.agent.map_height // 2, self.agent.map_height // 2)

        # Create occupancy map with ALL landmarks (once)
        occupancy_image = self._create_occupancy_map_image(landmarks, agent_map_pos)

        # Evaluate each landmark individually
        valid_landmarks = []
        invalid_landmarks = []

        for i, lm in enumerate(landmarks):
            landmark_id = lm['original_index']

            # Create prompt for single landmark evaluation
            prompt = f"""You are evaluating landmark {landmark_id} (blue square labeled "{landmark_id}").

Map shows:
- Blue squares with numbers: All candidate landmarks
- Yellow pixels: Frontiers (unexplored boundaries)
- Dark gray/black: Walls and obstacles
- White: Unknown area
- Red: Robot position

For landmark {landmark_id}:

Assume all other landmarks are kept.

Question:
If landmark {landmark_id} is removed, would the agent's future navigation or exploration decisions change in any meaningful way?

Answer ONLY:
- REMOVE (no meaningful change)
- KEEP (meaningful change)

Give one short reason."""

            # Get VLM response
            response = self.get_vlm_response(prompt, occupancy_image)

            # Parse response to determine if landmark should be kept
            should_keep, reason = self._parse_single_landmark_decision(response)

            if should_keep:
                valid_landmarks.append(lm)
            else:
                lm['filtering_reason'] = reason
                invalid_landmarks.append(lm)

            print(f"  Landmark {landmark_id}: {'KEEP' if should_keep else 'REMOVE'} - {reason}")

        filtered_count = len(invalid_landmarks)

        if filtered_count > 0:
            for lm in invalid_landmarks:
                pos = lm['position']
                reason = lm.get('filtering_reason', 'No reason provided')

        return valid_landmarks, filtered_count

    def _parse_single_landmark_decision(self, response):
        """
        Parse VLM response for a single landmark evaluation.

        Expected format:
        KEEP or REMOVE
        <reason>

        Args:
            response: VLM response string

        Returns:
            Tuple (should_keep: bool, reason: str)
        """
        import re

        response = response.strip()

        # Default values
        should_keep = True  # Conservative: keep by default if parsing fails
        reason = "No reason provided"

        # Check for KEEP or REMOVE in response
        response_upper = response.upper()

        if "REMOVE" in response_upper:
            should_keep = False
        elif "KEEP" in response_upper:
            should_keep = True

        # Extract reason (text after KEEP/REMOVE)
        lines = response.split('\n')
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            if "KEEP" in line_upper or "REMOVE" in line_upper:
                # Look for reason in subsequent lines
                if i + 1 < len(lines):
                    reason_lines = lines[i+1:]
                    reason = ' '.join([l.strip() for l in reason_lines if l.strip()])
                    if reason:
                        # Limit reason length
                        reason = reason[:100]
                    break

        # If no reason found, try to extract from same line
        if reason == "No reason provided":
            # Try to extract text after KEEP/REMOVE on same line
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.upper().startswith('KEEP'):
                    reason_match = re.search(r'KEEP\s*[:-]?\s*(.+)', line_stripped, re.IGNORECASE)
                    if reason_match:
                        reason = reason_match.group(1).strip()[:100]
                        break
                elif line_stripped.upper().startswith('REMOVE'):
                    reason_match = re.search(r'REMOVE\s*[:-]?\s*(.+)', line_stripped, re.IGNORECASE)
                    if reason_match:
                        reason = reason_match.group(1).strip()[:100]
                        break

        return should_keep, reason

    def _parse_vlm_filtering_response(self, response, num_landmarks):
        """
        Parse VLM response to extract invalid landmark indices and reasons.

        Supports two formats:
        1. JSON format (preferred): {"keep": [numbers], "remove": {"number": "reason"}}
        2. Legacy format: "- <number>: <reason>" (fallback)

        Args:
            response: VLM response string
            num_landmarks: Total number of landmarks

        Returns:
            Dictionary mapping invalid landmark numbers (0-indexed) to filtering reasons
        """
        import re
        import json

        invalid_landmarks = {}

        # Try to parse JSON format first (new format from GPT-5 prompt)
        try:
            # First, try to extract JSON from markdown code blocks
            json_str = None

            # Match ```json ... ``` or ``` ... ```
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # Try to find JSON directly in response (match complete JSON object)
                # Use a more robust pattern that handles nested braces
                brace_count = 0
                start_idx = -1
                for i, char in enumerate(response):
                    if char == '{':
                        if brace_count == 0:
                            start_idx = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_idx != -1:
                            # Found a complete JSON object
                            json_str = response[start_idx:i+1]
                            # Check if it contains "keep" and "remove"
                            if '"keep"' in json_str and '"remove"' in json_str:
                                break

            if json_str:
                data = json.loads(json_str)

                # Extract removed landmarks from "remove" field
                if "remove" in data and isinstance(data["remove"], dict):
                    for landmark_num_str, reason in data["remove"].items():
                        try:
                            idx = int(landmark_num_str)
                            if 0 <= idx < num_landmarks:
                                invalid_landmarks[idx] = reason
                        except (ValueError, TypeError):
                            continue

                # Successfully parsed JSON, return result
                print(f"[VLM Filter] Parsed JSON format: {len(invalid_landmarks)} landmarks to remove")
                return invalid_landmarks

        except (json.JSONDecodeError, AttributeError) as e:
            # JSON parsing failed, continue to fallback methods
            print(f"[VLM Filter] JSON parsing failed: {e}, trying fallback methods")
            pass

        # Check if all landmarks are valid
        if "all landmarks are valid" in response.lower():
            return invalid_landmarks

        # Fallback: Parse format: "- <number>: <reason>"
        # Match lines like "- 2: Located in a dead-end corner..."
        pattern = r'^[-*]\s*(\d+)\s*:\s*(.+)$'

        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(pattern, line)
            if match:
                try:
                    idx = int(match.group(1))
                    reason = match.group(2).strip()

                    # Validate index is in valid range (0 to num_landmarks-1)
                    if 0 <= idx < num_landmarks:
                        invalid_landmarks[idx] = reason
                except (ValueError, IndexError):
                    continue

        # Fallback: try to parse old format (Python list) for backward compatibility
        if len(invalid_landmarks) == 0:
            list_pattern = r'\[[\d,\s]*\]'
            matches = re.findall(list_pattern, response)

            if matches:
                try:
                    invalid_list = eval(matches[0])
                    if isinstance(invalid_list, list):
                        for idx in invalid_list:
                            if 0 <= idx < num_landmarks:
                                invalid_landmarks[idx] = "No reason provided (old format)"
                except:
                    pass

        # Final fallback: try to find numbers in the response
        if len(invalid_landmarks) == 0:
            numbers = re.findall(r'\b\d+\b', response)
            if numbers:
                for n in numbers:
                    idx = int(n)
                    if 0 <= idx < num_landmarks:
                        invalid_landmarks[idx] = "No reason provided (fallback parsing)"

        if len(invalid_landmarks) == 0 and "invalid" in response.lower():
            print(f"[Warning] Failed to parse VLM response: {response}")

        return invalid_landmarks

    def _filter_landmarks_by_llm(self, landmarks):
        """
        LLM-based filtering: Use LLM to judge landmark validity.

        Args:
            landmarks: List of landmarks with semantic information

        Returns:
            Tuple (valid_landmarks, filtered_count)
        """

        if not hasattr(self.agent, 'obj_goal_sg'):
            # If no goal specified, keep all landmarks
            return landmarks, 0

        goal_description = self.agent.obj_goal_sg

        # Create batch prompt with all landmarks
        prompt = self._create_batch_validity_prompt(landmarks, goal_description)

        # Get LLM response for all landmarks at once
        response = self.get_llm_response(prompt)

        # Parse response to get validity for each landmark
        validity_results = self._parse_batch_validity_response(response, len(landmarks))

        # Split landmarks into valid and invalid
        valid_landmarks = []
        invalid_landmarks = []

        for i, lm in enumerate(landmarks):
            if i < len(validity_results) and validity_results[i]:
                valid_landmarks.append(lm)
            else:
                invalid_landmarks.append(lm)

        filtered_count = len(invalid_landmarks)

        if filtered_count > 0:
            for lm in invalid_landmarks[:3]:  # Show first 3
                pos = lm['position']
                print(f"  - Landmark at ({self.map_height - pos[0]:.0f}, {pos[1]:.0f}): {lm['semantics']}")

        return valid_landmarks, filtered_count

    def _filter_landmarks_by_rules(self, landmarks):
        """
        Rule-based filtering: Filter landmarks based on info_gain ratio.

        Filtering rule:
        - Keep landmarks with (info_gain / max_possible_info) >= 4%
        - Remove landmarks with (info_gain / max_possible_info) < 4%

        Args:
            landmarks: List of landmarks with info_gain metrics

        Returns:
            Tuple (valid_landmarks, filtered_count)
        """

        valid_landmarks = []
        invalid_landmarks = []
        threshold = 0.04  # 4%

        for lm in landmarks:
            geom = lm['geometry']
            pos = lm['position']
            lm_id = lm['original_index']

            info_gain = geom.get('info_gain', 0)
            max_possible_info = geom.get('max_possible_info', 1)

            # Calculate info gain ratio
            info_gain_ratio = info_gain / max_possible_info if max_possible_info > 0 else 0

            # Apply threshold: keep if >= 4%
            is_valid = info_gain_ratio >= threshold

            # Format output
            pos_str = f"({self.map_height - pos[0]:.0f}, {pos[1]:.0f})"
            ratio_pct = info_gain_ratio * 100
            decision = "KEEP" if is_valid else "REMOVE"


            if is_valid:
                valid_landmarks.append(lm)
            else:
                invalid_landmarks.append(lm)


        filtered_count = len(invalid_landmarks)

        return valid_landmarks, filtered_count

    def _compute_environmental_semantic_entropy(self, landmarks, object_nodes, distance_threshold_meters=3.0):
        """
        Step 2.5: Compute environmental semantic entropy for each landmark.

        For each landmark, computes the sum of semantic entropy of all objects within
        distance_threshold_meters.
        The semantic entropy of an object O_i is defined as:
            H(O_i) = -Σ_c P(c|O_i) log P(c|O_i)
        where c is a class label and P(c|O_i) is the probability of object O_i being class c.

        Args:
            landmarks: List of valid landmarks with position information
            object_nodes: List of detected object nodes with caption probabilities
            distance_threshold_meters: Distance threshold in meters (default: 3.0m)
                                      Objects within this distance contribute to the entropy

        Returns:
            Updated landmarks with 'environmental_entropy' field added
        """

        for lm in landmarks:
            lm_pos_flipped = lm['position_flipped']  # [row, col] in map coordinates

            # Find all objects within distance_threshold_pixels
            nearby_objects = []
            total_entropy = 0.0

            for obj_node in object_nodes:
                if not hasattr(obj_node, 'center') or obj_node.center is None:
                    continue

                # obj_node.center is [x, y] = [col, row] in map coordinates
                obj_pos = np.array([obj_node.center[1], obj_node.center[0]])  # Convert to [row, col]

                # Calculate distance
                dist_pixels = np.linalg.norm(lm_pos_flipped - obj_pos)
                dist_meters = dist_pixels * self.map_resolution / 100.0

                if dist_meters <= distance_threshold_meters:
                    # Object is within range
                    # Compute semantic entropy for this object
                    entropy = self._compute_object_entropy(obj_node)

                    nearby_objects.append({
                        'caption': obj_node.caption,
                        'distance_meters': dist_meters,
                        'entropy': entropy
                    })
                    total_entropy += entropy

            # Store environmental entropy
            lm['geometry']['environmental_entropy'] = total_entropy
            lm['geometry']['nearby_object_count'] = len(nearby_objects)
            lm['geometry']['nearby_objects'] = nearby_objects

        return landmarks

    def _compute_object_entropy(self, obj_node):
        """
        Compute semantic entropy for a single object.

        H(O) = -Σ_c P(c|O) log P(c|O)

        Args:
            obj_node: Object node with caption probability distribution

        Returns:
            float: Semantic entropy value
        """
        # Check if object has probability distribution
        if not hasattr(obj_node, 'object') or obj_node.object is None:
            return 0.0

        obj_data = obj_node.object

        # Get caption probabilities
        if 'caption_probs' not in obj_data or 'captions_sorted' not in obj_data:
            # No distribution available, assume deterministic (entropy = 0)
            return 0.0

        caption_probs = obj_data['caption_probs']
        captions_sorted = obj_data['captions_sorted']

        # Compute entropy
        entropy = 0.0
        for caption in captions_sorted:
            prob = caption_probs.get(caption, 0.0)
            if prob > 0:
                entropy -= prob * np.log(prob)

        return entropy

    def _create_batch_validity_prompt(self, landmarks, goal_description):
        """
        Create a batch prompt for LLM to judge all landmarks at once.

        Args:
            landmarks: List of landmark dictionaries
            goal_description: Description of the goal object

        Returns:
            str: Formatted batch prompt for LLM
        """
        prompt_header = f"""Task: The robot is searching for "{goal_description}". Judge which landmarks have exploration potential.

A landmark has exploration potential if it can expand the explored map and help discover new areas.
A landmark has NO potential if it's in a heavily explored area (>95%) AND far from unexplored frontiers.

Landmarks to evaluate:
"""

        # Add each landmark's information
        landmark_descriptions = []
        for i, lm in enumerate(landmarks):
            geom = lm['geometry']
            semantics = lm['semantics']

            # Build exploration status
            if geom['explored_fully']:
                exploration_status = "heavily explored (>95%)"
            elif geom['exploration_pct'] < 0.3:
                exploration_status = "mostly unexplored (<30%)"
            else:
                exploration_status = f"{geom['exploration_pct']*100:.0f}% explored"

            frontier_status = "near frontiers" if geom['near_frontier'] else "far from frontiers"

            # Format room information
            if geom['room_name'] == 'unknown' and geom.get('nearest_room_info'):
                nearest = geom['nearest_room_info']
                room_info = f"nearest to {nearest['room_name']} ({nearest['distance_meters']:.1f}m away)"
            else:
                room_info = f"{geom['room_name']} ({geom['room_probability']:.0%})"

            lm_desc = f"""
{i+1}. {semantics}
   - Distance: {geom['distance_to_agent_meters']:.1f}m
   - Surrounding: {exploration_status}
   - Frontier: {frontier_status}
   - Room: {room_info}"""

            landmark_descriptions.append(lm_desc)

        prompt_body = "\n".join(landmark_descriptions)

        prompt_footer = """

Answer format (return ONLY the numbers and YES/NO, one per line):
1: YES/NO
2: YES/NO
3: YES/NO
...

Example:
1: YES
2: NO
3: YES"""

        return prompt_header + prompt_body + prompt_footer

    def _parse_batch_validity_response(self, response, num_landmarks):
        """
        Parse LLM batch response to extract validity for each landmark.

        Args:
            response: LLM response string
            num_landmarks: Expected number of landmarks

        Returns:
            list: Boolean list indicating validity for each landmark
        """
        # Initialize all as valid (conservative default)
        validity_results = [True] * num_landmarks

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse format: "1: YES" or "1: NO"
            if ':' in line:
                parts = line.split(':', 1)
                try:
                    idx = int(parts[0].strip()) - 1  # Convert to 0-indexed
                    answer = parts[1].strip().upper()

                    if 0 <= idx < num_landmarks:
                        if answer.startswith('YES'):
                            validity_results[idx] = True
                        elif answer.startswith('NO'):
                            validity_results[idx] = False
                except (ValueError, IndexError):
                    continue

        return validity_results

    def _compute_landmark_goal_prior_probabilities(self, landmarks, goal_description, scenegraph_combinations):
        """
        Step 3: Compute prior probability of finding goal for each landmark.

        The prior probability is computed as a normalized combination of:
        1. Environmental semantic entropy (50% weight)
        2. Geometric information gain ratio (50% weight)

        Higher entropy and higher info gain ratio indicate better exploration potential.

        Args:
            landmarks: List of valid landmarks
            goal_description: Description of goal object
            scenegraph_combinations: Scene graph combinations for context

        Returns:
            Updated landmarks with prior_probability field filled
        """

        if len(landmarks) == 0:
            return landmarks

        # Extract metrics for all landmarks
        env_entropies = []
        info_gain_ratios = []

        for lm in landmarks:
            geom = lm['geometry']

            # Get environmental entropy
            env_entropy = geom.get('environmental_entropy', 0.0)
            env_entropies.append(env_entropy)

            # Get info gain ratio
            info_gain = geom.get('info_gain', 0)
            max_possible_info = geom.get('max_possible_info', 1)
            info_gain_ratio = info_gain / max_possible_info if max_possible_info > 0 else 0.0
            info_gain_ratios.append(info_gain_ratio)

        # Convert to numpy arrays for easier computation
        env_entropies = np.array(env_entropies)
        info_gain_ratios = np.array(info_gain_ratios)

        # Normalize environmental entropy to [0, 1]
        # Higher entropy is better (more uncertainty = more exploration potential)
        if env_entropies.max() > 0:
            env_entropy_normalized = env_entropies / env_entropies.max()
        else:
            env_entropy_normalized = np.zeros_like(env_entropies)

        # Normalize info gain ratio to [0, 1]
        # Higher ratio is better (more unknown area visible)
        if info_gain_ratios.max() > 0:
            info_gain_normalized = info_gain_ratios / info_gain_ratios.max()
        else:
            info_gain_normalized = np.zeros_like(info_gain_ratios)

        prior_probabilities = 0.1 * env_entropy_normalized + 0.9 * info_gain_normalized

        # Print detailed table

        for i, lm in enumerate(landmarks):
            lm_id = lm['original_index']
            pos_str = f"({self.map_height - lm['position'][0]:.0f}, {lm['position'][1]:.0f})"

            # Store prior probability
            lm['prior_probability'] = float(prior_probabilities[i])

            # Print row
            #       f"{env_entropies[i]:<15.4f} "
            #       f"{info_gain_ratios[i]:<15.4f} "
            #       f"{env_entropy_normalized[i]:<15.4f} "
            #       f"{info_gain_normalized[i]:<15.4f} "
            #       f"{prior_probabilities[i]:<12.4f}")


        # # Print summary statistics

        return landmarks

    def _compare_landmarks_pairwise(self, landmarks, scenegraph_combinations, goal_description):
        """
        Step 4: Perform pairwise comparisons between landmarks using LLM.

        For each pair of landmarks:
        - Compare them K times (allowing same scene graph to be sampled)
        - For each comparison:
          * Sample a scene graph combination (with replacement)
          * Assign scene graph semantics to both landmarks
          * Ask LLM which landmark is more likely to lead to goal
          * Record the winner

        Args:
            landmarks: List of valid landmarks
            scenegraph_combinations: Available scene graph combinations
            goal_description: Goal object description

        Returns:
            Dictionary containing:
            - 'comparisons': List of comparison records
            - 'win_counts': Dict mapping landmark index to win count
            - 'total_comparisons': Total number of comparisons performed
        """

        n = len(landmarks)
        K = 3  # Number of comparisons per pair

        comparison_results = {
            'comparisons': [],
            'win_counts': {i: 0 for i in range(n)},
            'total_comparisons': 0
        }

        # Compare all pairs
        total_pairs = n * (n - 1) // 2

        for i in range(n):
            for j in range(i + 1, n):
                # Compare this pair K times
                for round_idx in range(K):
                    # 1. Sample a scene graph combination
                    sampled_sg = self._sample_scenegraph(scenegraph_combinations)

                    # 2. Add scene graph semantics to both landmarks
                    lm_i_enriched = self._add_scenegraph_semantics_to_landmark(
                        landmarks[i], sampled_sg
                    )
                    lm_j_enriched = self._add_scenegraph_semantics_to_landmark(
                        landmarks[j], sampled_sg
                    )

                    # 3. Create comparison prompt
                    prompt = self._create_pairwise_comparison_prompt(
                        lm_i_enriched, lm_j_enriched, goal_description, i, j
                    )

                    # 4. Get LLM response
                    response = self.get_llm_response(prompt)

                    # 5. Parse winner
                    winner = self._parse_comparison_winner(response, i, j)

                    # 6. Record results
                    landmarks[winner]['win_count'] += 1
                    landmarks[i]['total_comparisons'] += 1
                    landmarks[j]['total_comparisons'] += 1

                    comparison_results['comparisons'].append({
                        'landmark_i': i,
                        'landmark_j': j,
                        'winner': winner,
                        'scenegraph_id': sampled_sg.get('id', 'unknown') if sampled_sg else None,
                        'round': round_idx
                    })
                    comparison_results['win_counts'][winner] += 1
                    comparison_results['total_comparisons'] += 1


        return comparison_results

    def _create_comparison_occupancy_map(self, comparison_landmarks, agent_map_pos):
        """
        Create an occupancy map visualization showing two landmarks being compared.

        Args:
            comparison_landmarks: List of 2 landmarks to compare
            agent_map_pos: Agent position in map coordinates

        Returns:
            PIL Image: Occupancy map with landmarks A and B marked
        """
        import torch
        import skimage.morphology
        from matplotlib import colors
        import cv2
        from scipy.ndimage import binary_dilation

        # Get map dimensions
        H, W = self.agent.full_map.shape[-2:]

        # Create base map (same as filtering visualization)
        paper_map_trans = torch.ones((H, W, 3), dtype=torch.float32)
        unknown_rgb = colors.to_rgb('#FFFFFF')
        paper_map_trans[:, :, :] = torch.tensor(unknown_rgb, dtype=torch.float32)

        # Draw free space (light gray)
        free_rgb = colors.to_rgb('#E7E7E7')
        free_mask = self.agent.fbe_free_map.cpu().numpy()[0, 0, ::-1] > 0.5
        paper_map_trans[free_mask, :] = torch.tensor(free_rgb, dtype=torch.float32)

        # Draw obstacles (darker gray)
        obstacle_rgb = colors.to_rgb('#A2A2A2')
        obs_mask = skimage.morphology.binary_dilation(
            self.agent.full_map.cpu().numpy()[0, 0, ::-1] > 0.5,
            skimage.morphology.disk(1)
        )
        paper_map_trans[obs_mask, :] = torch.tensor(obstacle_rgb, dtype=torch.float32)

        # Convert to channel-first format
        paper_map_trans_chw = paper_map_trans.permute(2, 0, 1)

        # Calculate and draw frontiers (yellow)
        from utils.utils_frontiers import calculate_frontiers
        frontier_map, frontier_locations, num_frontiers = calculate_frontiers(
            self.agent.full_map, self.agent.fbe_free_map
        )

        # Draw frontiers before agent (so agent is on top)
        if frontier_locations is not None and len(frontier_locations) > 0:
            frontier_color = (1.0, 1.0, 0.0)  # Yellow
            frontier_size = 1

            # Convert frontier locations to numpy if needed
            if torch.is_tensor(frontier_locations):
                fl_np = frontier_locations.cpu().numpy()
            else:
                fl_np = frontier_locations

            # Transform frontier coordinates
            for frontier in fl_np:
                try:
                    r = int(frontier[0])
                    c = int(frontier[1])
                except Exception:
                    continue

                # Use same coordinate transform as landmarks
                top = int(self.agent.map_size_cm/5) - r - frontier_size
                bottom = int(self.agent.map_size_cm/5) - r + frontier_size
                left = c - frontier_size
                right = c + frontier_size

                # Clip to bounds
                H_ch, W_ch = paper_map_trans_chw.shape[1], paper_map_trans_chw.shape[2]
                top = max(0, top)
                left = max(0, left)
                bottom = min(H_ch, bottom)
                right = min(W_ch, right)

                if top >= bottom or left >= right:
                    continue

                # Draw yellow frontier
                paper_map_trans_chw[0, top:bottom, left:right] = frontier_color[0]
                paper_map_trans_chw[1, top:bottom, left:right] = frontier_color[1]
                paper_map_trans_chw[2, top:bottom, left:right] = frontier_color[2]

        # Draw agent position (on top of frontiers)
        from utils.image_process import draw_agent
        for idx, pose in enumerate(self.agent.history_pose):
            draw_step_num = 30
            alpha = max(0, 1 - (len(self.agent.history_pose) - idx) / draw_step_num)
            agent_size = 1
            if idx == len(self.agent.history_pose) - 1:
                agent_size = 2
            draw_agent(agent=self.agent, map=paper_map_trans_chw, pose=pose,
                      agent_size=agent_size, color_index=0, alpha=alpha)

        # Convert to RGB numpy array
        occupancy_map_full = (paper_map_trans_chw.permute(1, 2, 0) * 255).numpy().astype(np.uint8)

        # Overlay agent trajectory history
        if hasattr(self.agent, 'agent_trajectory_map') and self.agent.agent_trajectory_map is not None:
            trajectory_map = self.agent.agent_trajectory_map
            H_traj, W_traj = trajectory_map.shape
            H_map, W_map = occupancy_map_full.shape[:2]

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
            expanded_trajectory = binary_dilation(trajectory_map > 0, structure=kernel).astype(np.uint8)

            light_green = np.array([150, 255, 150], dtype=np.uint8)
            alpha_light = 0.2
            medium_green = np.array([100, 220, 100], dtype=np.uint8)
            alpha_core = 0.35

            mask_expanded = expanded_trajectory > 0
            if H_traj == H_map and W_traj == W_map:
                occupancy_map_full[mask_expanded] = (
                    occupancy_map_full[mask_expanded] * (1 - alpha_light) +
                    light_green * alpha_light
                ).astype(np.uint8)

                mask_core = trajectory_map > 0
                occupancy_map_full[mask_core] = (
                    occupancy_map_full[mask_core] * (1 - alpha_core) +
                    medium_green * alpha_core
                ).astype(np.uint8)

        # Draw the two comparison landmarks (A and B)
        def transform_coords(r, c):
            y = int(int(self.agent.map_size_cm/5) - int(r))
            x = int(c)
            return (x, y)

        H_map, W_map = occupancy_map_full.shape[:2]
        landmark_size = 4

        labels = ['A', 'B']
        colors_list = [(255, 0, 255), (0, 255, 255)]  # Magenta for A, Cyan for B

        for idx, (lm, label, color) in enumerate(zip(comparison_landmarks, labels, colors_list)):
            pos = lm['position']
            r, c = int(pos[0]), int(pos[1])
            lm_x, lm_y = transform_coords(r, c)

            # Draw colored square
            top = lm_y - landmark_size
            bottom = lm_y + landmark_size
            left = lm_x - landmark_size
            right = lm_x + landmark_size

            top = max(0, top)
            left = max(0, left)
            bottom = min(H_map, bottom)
            right = min(W_map, right)

            if top < bottom and left < right:
                occupancy_map_full[top:bottom, left:right] = color

                # Draw label next to landmark
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2

                text_x = right + 3
                text_y = lm_y + 3

                if text_x < W_map - 15:
                    # White background
                    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    cv2.rectangle(occupancy_map_full,
                                (text_x - 2, text_y - text_h - 2),
                                (text_x + text_w + 2, text_y + 3),
                                (255, 255, 255), -1)
                    # Black text
                    cv2.putText(occupancy_map_full, label, (text_x, text_y),
                              font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Convert to PIL Image
        image = Image.fromarray(occupancy_map_full)
        return image

    def _create_pairwise_comparison_prompt_vlm(self, landmark_a, landmark_b, goal_description):
        """
        Create a prompt for VLM to compare two landmarks using occupancy map.

        Args:
            landmark_a: First landmark (enriched with scene graph)
            landmark_b: Second landmark (enriched with scene graph)
            goal_description: Goal object description

        Returns:
            str: Formatted comparison prompt for VLM
        """
        geom_a = landmark_a['geometry']
        geom_b = landmark_b['geometry']

        # Format room information
        if geom_a['room_name'] == 'unknown' and geom_a.get('nearest_room_info'):
            nearest_a = geom_a['nearest_room_info']
            room_info_a = f"nearest to {nearest_a['room_name']} ({nearest_a['distance_meters']:.1f}m away)"
        else:
            room_info_a = f"{geom_a['room_name']} ({geom_a['room_probability']:.0%})"

        if geom_b['room_name'] == 'unknown' and geom_b.get('nearest_room_info'):
            nearest_b = geom_b['nearest_room_info']
            room_info_b = f"nearest to {nearest_b['room_name']} ({nearest_b['distance_meters']:.1f}m away)"
        else:
            room_info_b = f"{geom_b['room_name']} ({geom_b['room_probability']:.0%})"

        # Generate exploration context
        exploration_context = self._get_exploration_context(goal_description)

        prompt = f"""You are analyzing a robot navigation map to help find "{goal_description}". The image shows:
- White areas: explored free space
- Gray areas: obstacles/walls
- Light gray areas: unexplored regions
- Yellow dots: exploration frontiers (boundaries between explored and unexplored areas)
- Red circle: current robot position
- Magenta square (A): Landmark A
- Cyan square (B): Landmark B

{exploration_context}

Landmark A:
- Location: {landmark_a['semantics']}
- Distance from robot: {geom_a['distance_to_agent_meters']:.1f}m
- Surrounding: {geom_a['exploration_pct']*100:.0f}% explored, {"near frontiers" if geom_a['near_frontier'] else "far from frontiers"}
- Room: {room_info_a}
- Nearby objects: {landmark_a.get('scene_description', 'none')}

Landmark B:
- Location: {landmark_b['semantics']}
- Distance from robot: {geom_b['distance_to_agent_meters']:.1f}m
- Surrounding: {geom_b['exploration_pct']*100:.0f}% explored, {"near frontiers" if geom_b['near_frontier'] else "far from frontiers"}
- Room: {room_info_b}
- Nearby objects: {landmark_b.get('scene_description', 'none')}

Question: Based on the occupancy map and the information above, which landmark (A or B) is MORE likely to help find "{goal_description}"?

Consider:
1. Spatial layout: Which landmark has better access to unexplored areas (yellow frontiers)?
2. Room context: Which location is more likely to contain {goal_description}?
3. Navigation efficiency: Which provides better exploration potential?
4. Visual context: What does the occupancy map reveal about each landmark's surroundings?

Answer with ONLY "A" or "B" and a brief reason (one sentence).
Format: A/B: [reason]"""

        return prompt

    def _sample_scenegraph(self, scenegraph_combinations):
        """
        Sample a scene graph combination from the available combinations.

        Args:
            scenegraph_combinations: SceneGraphCombinations object

        Returns:
            dict: Sampled scene graph combination
        """
        if scenegraph_combinations is None or len(scenegraph_combinations.all_combinations) == 0:
            return None

        # Sample based on probabilities
        combinations = scenegraph_combinations.all_combinations
        if len(combinations) == 0:
            return None

        # Get probabilities
        probs = np.array([comb.get('probability', 1.0) for comb in combinations])
        probs = probs / probs.sum()  # Normalize

        # Sample with replacement
        idx = np.random.choice(len(combinations), p=probs)
        return combinations[idx]

    def _add_scenegraph_semantics_to_landmark(self, landmark, scenegraph, distance_threshold_meters=3.0):
        """
        Add scene graph semantics to a landmark (objects near it).

        Uses the nearby_objects already computed in environmental semantic entropy step,
        which includes distance filtering.

        Args:
            landmark: Landmark dictionary with geometry and semantics
            scenegraph: Sampled scene graph combination (for compatibility, not used)
            distance_threshold_meters: Distance threshold in meters (default: 3.0m)
                                      This should match the threshold used in
                                      _compute_environmental_semantic_entropy

        Returns:
            dict: Enriched landmark with scene graph information
        """
        enriched = landmark.copy()

        # Check if nearby_objects already computed (from environmental entropy step)
        if 'nearby_objects' in landmark['geometry'] and len(landmark['geometry']['nearby_objects']) > 0:
            # Use pre-computed nearby objects (already filtered by distance)
            nearby_objects = landmark['geometry']['nearby_objects']

            # Extract just the captions for description
            object_captions = [obj['caption'] for obj in nearby_objects]
            enriched['nearby_objects'] = object_captions

            # Generate scene description with distance info
            if len(nearby_objects) > 0:
                # Sort by distance (closest first) and take top 5
                sorted_objects = sorted(nearby_objects, key=lambda x: x['distance_meters'])[:5]
                obj_descriptions = [
                    f"{obj['caption']} ({obj['distance_meters']:.1f}m)"
                    for obj in sorted_objects
                ]
                enriched['scene_description'] = f"near {', '.join(obj_descriptions)}"
            else:
                enriched['scene_description'] = "no detected objects nearby"
        else:
            # Fallback: no nearby objects
            enriched['nearby_objects'] = []
            enriched['scene_description'] = "no detected objects nearby"

        return enriched

    def _create_pairwise_comparison_prompt(self, landmark_a, landmark_b, goal_description, idx_a, idx_b):
        """
        Create a prompt for LLM to compare two landmarks.

        Args:
            landmark_a: First landmark (enriched with scene graph)
            landmark_b: Second landmark (enriched with scene graph)
            goal_description: Goal object description
            idx_a: Index of landmark A
            idx_b: Index of landmark B

        Returns:
            str: Formatted comparison prompt
        """
        geom_a = landmark_a['geometry']
        geom_b = landmark_b['geometry']

        # Format room information for landmark A
        if geom_a['room_name'] == 'unknown' and geom_a.get('nearest_room_info'):
            nearest_a = geom_a['nearest_room_info']
            room_info_a = f"nearest to {nearest_a['room_name']} ({nearest_a['distance_meters']:.1f}m away)"
        else:
            room_info_a = f"{geom_a['room_name']} ({geom_a['room_probability']:.0%})"

        # Format room information for landmark B
        if geom_b['room_name'] == 'unknown' and geom_b.get('nearest_room_info'):
            nearest_b = geom_b['nearest_room_info']
            room_info_b = f"nearest to {nearest_b['room_name']} ({nearest_b['distance_meters']:.1f}m away)"
        else:
            room_info_b = f"{geom_b['room_name']} ({geom_b['room_probability']:.0%})"

        # Generate exploration context
        exploration_context = self._get_exploration_context(goal_description)

        prompt = f"""Task: The robot is searching for "{goal_description}". Compare two landmarks and decide which one is more likely to help find the goal.

{exploration_context}
Landmark A:
- Location: {landmark_a['semantics']}
- Distance from robot: {geom_a['distance_to_agent_meters']:.1f}m
- Surrounding: {geom_a['exploration_pct']*100:.0f}% explored, {"near frontiers" if geom_a['near_frontier'] else "far from frontiers"}
- Room: {room_info_a}
- Nearby objects: {landmark_a.get('scene_description', 'none')}

Landmark B:
- Location: {landmark_b['semantics']}
- Distance from robot: {geom_b['distance_to_agent_meters']:.1f}m
- Surrounding: {geom_b['exploration_pct']*100:.0f}% explored, {"near frontiers" if geom_b['near_frontier'] else "far from frontiers"}
- Room: {room_info_b}
- Nearby objects: {landmark_b.get('scene_description', 'none')}

Question: Which landmark is MORE likely to help find "{goal_description}"?

Consider:
1. Which location is more likely to have {goal_description} based on room type and nearby objects?
2. Which has better exploration potential (unexplored areas, near frontiers)?
3. Which provides better information gain for finding the goal?

Answer with ONLY "A" or "B" and a brief reason (one sentence).
Format: A/B: [reason]"""

        return prompt

    def _parse_comparison_winner(self, response, idx_a, idx_b):
        """
        Parse LLM response to determine the winner of a pairwise comparison.

        Args:
            response: LLM response string
            idx_a: Index of landmark A
            idx_b: Index of landmark B

        Returns:
            int: Index of winning landmark (idx_a or idx_b)
        """
        response_upper = response.upper().strip()

        # Check for A or B at the beginning
        if response_upper.startswith('A'):
            return idx_a
        elif response_upper.startswith('B'):
            return idx_b

        # Fallback: search for A or B in the response
        # Count occurrences to break ties
        count_a = response_upper.count('LANDMARK A')
        count_b = response_upper.count('LANDMARK B')

        if count_a > count_b:
            return idx_a
        elif count_b > count_a:
            return idx_b

        # If still unclear, default to first landmark (conservative)
        print(f"[Warning] Unclear comparison response: {response[:100]}")
        return idx_a

    def _compute_relative_expected_utility(self, landmarks, comparison_results):
        """
        Step 5: Compute Relative Expected Utility for each landmark.

        For each valid landmark L_i, its utility U(L_i) is defined as the weighted
        probability that it beats all other landmarks across all probabilistic universes.

        The utility is computed based on:
        - Win/loss record in pairwise comparisons
        - Prior probability of finding goal
        - Bradley-Terry style strength estimation

        Args:
            landmarks: List of landmarks with comparison results
            comparison_results: Results from pairwise comparisons

        Returns:
            Tuple (best_landmark_idx, best_utility, all_utilities)
        """

        n = len(landmarks)
        utilities = []

        for i, lm in enumerate(landmarks):
            # 1. Compute win rate from comparisons
            if lm['total_comparisons'] > 0:
                win_rate = lm['win_count'] / lm['total_comparisons']
            else:
                # No comparisons (only 1 landmark) - use neutral value
                win_rate = 0.5

            # 2. Get prior probability
            prior = lm.get('prior_probability', 0.5)

            # 3. Compute utility as weighted combination
            # Give more weight to comparison results (70%) than prior (30%)
            # because comparisons contain more information
            utility = 0.7 * win_rate + 0.3 * prior

            utilities.append(utility)

            print(f"  Landmark {i}: win_rate={win_rate:.3f}, prior={prior:.3f}, utility={utility:.3f}")

        # Select landmark with highest utility
        best_idx = int(np.argmax(utilities))
        best_utility = utilities[best_idx]


        return best_idx, best_utility, utilities

    def _generate_comparison_explanation(self, landmark, utility, comparison_results):
        """Generate human-readable explanation for the selected landmark."""
        pos = landmark['position']
        win_count = landmark['win_count']
        total_comp = landmark['total_comparisons']
        geom = landmark['geometry']

        explanation = (
            f"Selected Landmark at ({self.map_height - pos[0]:.0f}, {pos[1]:.0f}) "
            f"with utility {utility:.3f}. "
        )

        # Add win rate information
        if total_comp > 0:
            win_rate = win_count / total_comp
            explanation += f"Won {win_count}/{total_comp} comparisons ({win_rate:.1%}). "

        # Add spatial information
        explanation += f"Location: {landmark['semantics']}. "

        # Add exploration status
        if geom['near_frontier']:
            explanation += "Near unexplored frontiers. "
        if geom['explored_fully']:
            explanation += "Area heavily explored. "

        return explanation

    def _compute_exploration_status(self, row, col, radius=40):
        """
        Compute exploration status around a landmark position using raycasting-based information gain.

        Instead of checking explored percentage, this method:
        1. Simulates standing at the landmark position
        2. Casts rays in all directions (like a lidar)
        3. Counts how many unknown pixels can be seen (information gain)
        4. Rays stop at obstacles and have max range of 2m

        Args:
            row, col: Landmark position in map coordinates
            radius: Deprecated, kept for compatibility (using 2m max_range instead)
        Returns:
            dict: {
                'exploration_pct': normalized information gain (0-1),
                'explored_fully': bool (info_gain below threshold),
                'info_gain': raw count of visible unknown pixels,
                'max_possible_info': maximum possible info gain
            }
        """
        # Use raycasting to compute information gain
        info_gain, max_possible_info = self._compute_information_gain_raycasting(row, col)

        # Normalize to 0-1 (for compatibility with existing code)
        # Higher info_gain means LESS explored (more unknown area visible)
        # So exploration_pct = 1 - (info_gain / max_possible_info)
        if max_possible_info > 0:
            exploration_pct = 1.0 - (info_gain / max_possible_info)
        else:
            exploration_pct = 1.0  # No rays hit anything, consider fully explored

        # Consider fully explored if info gain is very low
        # Threshold: if less than 2.5% of max possible info gain is visible
        info_gain_threshold = 0.045
        explored_fully = (info_gain / max_possible_info) < info_gain_threshold if max_possible_info > 0 else True

        return {
            'exploration_pct': exploration_pct,
            'explored_fully': explored_fully,
            'info_gain': info_gain,
            'max_possible_info': max_possible_info
        }

    def _compute_information_gain_raycasting(self, row, col, max_range_meters=4.0, num_rays=72):
        """
        Compute information gain using raycasting from a landmark position.

        Simulates standing at (row, col) and casting rays in all directions.
        Counts how many unknown (unexplored) pixels are visible.

        Args:
            row, col: Landmark position in map coordinates
            max_range_meters: Maximum ray distance (default: 2.0m)
            num_rays: Number of rays to cast (default: 36, every 10 degrees)
        Returns:
            tuple: (info_gain, max_possible_info)
                - info_gain: number of unique unknown pixels visible
                - max_possible_info: maximum possible unknown pixels (for normalization)
        """
        if not hasattr(self.agent, 'fbe_free_map') or not hasattr(self.agent, 'full_map'):
            return 0, 1  # No maps available

        # Get maps
        free_map = self.agent.fbe_free_map.cpu().numpy()
        obstacle_map = self.agent.full_map.cpu().numpy()

        # Handle shape
        if len(free_map.shape) == 4:
            free_map_2d = free_map[0, 0, ::-1]  # Flip for correct orientation
            obstacle_map_2d = obstacle_map[0, 0, ::-1]  # Flip for correct orientation
        else:
            free_map_2d = free_map[::-1]
            obstacle_map_2d = obstacle_map[::-1]

        h, w = free_map_2d.shape

        # Convert max_range to pixels
        max_range_pixels = int(max_range_meters * 100.0 / self.map_resolution)

        # Track unique unknown pixels seen (use set to avoid double counting)
        unknown_pixels_seen = set()

        # For visualization: store ray endpoints and unknown pixels
        ray_endpoints = []
        hit_obstacles = []

        # Cast rays in all directions
        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays
            dx = np.cos(angle)
            dy = np.sin(angle)

            ray_stopped_at = None
            ray_hit_obstacle = False

            # Cast this ray
            for step in range(1, max_range_pixels + 1):
                # Current position along ray
                ray_row = int(row + dy * step)
                ray_col = int(col + dx * step)

                # Check bounds
                if ray_row < 0 or ray_row >= h or ray_col < 0 or ray_col >= w:
                    ray_stopped_at = (ray_row, ray_col)
                    break

                # Check if hit obstacle (stop ray)
                if obstacle_map_2d[ray_row, ray_col] > 0.5:
                    ray_stopped_at = (ray_row, ray_col)
                    ray_hit_obstacle = True
                    break

                # Check if this pixel is unknown (unexplored)
                # Unknown = not in free_map AND not obstacle
                is_free = free_map_2d[ray_row, ray_col] > 0.5
                is_obstacle = obstacle_map_2d[ray_row, ray_col] > 0.5

                if not is_free and not is_obstacle:
                    # This is an unknown pixel!
                    unknown_pixels_seen.add((ray_row, ray_col))

                # If this is the last step, record endpoint
                if step == max_range_pixels:
                    ray_stopped_at = (ray_row, ray_col)

            if ray_stopped_at:
                ray_endpoints.append(ray_stopped_at)
                if ray_hit_obstacle:
                    hit_obstacles.append(ray_stopped_at)

        info_gain = len(unknown_pixels_seen)

        # Calculate max possible info gain (for normalization)
        max_possible_info = int(np.pi * max_range_pixels ** 2)

        return info_gain, max_possible_info

    def _generate_landmark_description(self, dist_meters, near_frontier,
                                      frontier_dist_meters, exploration_status,
                                      room_name, room_prob):
        """
        Generate concise natural language description for a landmark.

        Args:
            dist_meters: Distance to agent in meters
            near_frontier: Boolean, if landmark is near unexplored area
            frontier_dist_meters: Distance to nearest frontier in meters
            exploration_status: Dict with exploration_pct and explored_fully
            room_name: Name of room landmark is in
            room_prob: Probability of room assignment

        Returns:
            str: Natural language description
        """
        parts = []

        # Distance description
        parts.append(f"{dist_meters:.1f}m away")

        # Room information
        if room_name != "unknown" and room_prob > 0.3:
            parts.append(f"in {room_name}")

        # Frontier information
        if near_frontier:
            parts.append("near unexplored area")

        # Exploration status
        exp_pct = exploration_status['exploration_pct']
        if exploration_status['explored_fully']:
            parts.append("heavily explored")
        elif exp_pct < 0.3:
            parts.append("mostly unexplored")
        elif exp_pct < 0.7:
            parts.append("partially explored")

        return ", ".join(parts)

    def _get_overall_exploration_stats(self):
        """
        Compute overall exploration statistics from fbe_free_map.

        Directly counts explored area in square meters without using obstacle map,
        since obstacle map includes exterior void areas that are not part of the building.

        Returns:
            dict: Contains explored_area_sqm (explored area in square meters)
                  Returns None if map is not available
        """
        if not hasattr(self.agent, 'fbe_free_map') or self.agent.fbe_free_map is None:
            return None

        free_map = self.agent.fbe_free_map
        # Convert to numpy if it's a tensor
        if hasattr(free_map, 'cpu'):
            free_map = free_map.cpu().numpy()

        # Handle shape - could be (1, 1, h, w) or (h, w)
        if len(free_map.shape) == 4:
            free_map_2d = free_map[0, 0]
        elif len(free_map.shape) == 2:
            free_map_2d = free_map
        else:
            return None

        # Count explored cells (where fbe_free_map == 1)
        explored_cells = np.sum(free_map_2d == 1)

        # Convert to square meters
        # Each cell represents (map_resolution cm)^2
        # map_resolution is in cm, so we need to convert to meters
        cell_area_sqm = (self.map_resolution / 100.0) ** 2  # Convert cm to m
        explored_area_sqm = explored_cells * cell_area_sqm

        return {
            'explored_area_sqm': explored_area_sqm,
            'explored_cells': int(explored_cells),
            'cell_area_sqm': cell_area_sqm
        }

    def _analyze_room_coverage(self):
        """
        Analyze coverage and exploration status for each detected room.

        Returns:
            list: List of room info dictionaries, each containing:
                  - name: room name
                  - confidence: average probability in detected area
                  - area_pixels: total detected area
                  - coverage: exploration percentage within this room
                  - high_conf_area: area with high confidence (>0.5)
        """
        if not hasattr(self.agent, 'room_map') or self.agent.room_map is None:
            return []

        room_info = []

        for room_idx in range(1, self.agent.room_map.shape[1]):  # Skip index 0 (unknown)
            room_channel = self.agent.room_map[0, room_idx, :, :].cpu().numpy()
            room_name = self.agent.rooms[room_idx]

            # Calculate statistics for this room
            high_conf_area = np.sum(room_channel > 0.5)  # High confidence area
            med_conf_area = np.sum((room_channel > 0.1) & (room_channel <= 0.5))  # Medium confidence

            # Skip if room has very small detected area
            if high_conf_area + med_conf_area < 50:  # Less than 50 pixels
                continue

            # Calculate average confidence in detected area
            detected_mask = (room_channel > 0.1)
            avg_prob = room_channel[detected_mask].mean() if np.any(detected_mask) else 0

            # Calculate exploration coverage within this room
            # How much of the detected room area has been explored
            coverage = 0.0
            if hasattr(self.agent, 'fbe_free_map') and self.agent.fbe_free_map is not None:
                free_map = self.agent.fbe_free_map.cpu().numpy()
                if len(free_map.shape) == 4:
                    free_map_2d = free_map[0, 0]
                else:
                    free_map_2d = free_map

                explored_in_room = np.sum((free_map_2d == 1) & detected_mask)
                total_room_area = np.sum(detected_mask)
                coverage = explored_in_room / total_room_area if total_room_area > 0 else 0.0

            room_info.append({
                'name': room_name,
                'confidence': float(avg_prob),
                'area_pixels': int(high_conf_area + med_conf_area),
                'coverage': float(coverage),
                'high_conf_area': int(high_conf_area)
            })

        return room_info

    def _compute_goal_room_relevance(self, goal_description, room_name):
        """
        Compute relevance score between a goal object and a room type.

        Uses common-sense knowledge about where objects are typically found.

        Args:
            goal_description: Name of the goal object (e.g., "toilet", "bed")
            room_name: Name of the room (e.g., "bathroom", "bedroom")

        Returns:
            float: Relevance score between 0.0 and 1.0
        """
        # Normalize strings to lowercase for matching
        goal = goal_description.lower().strip()
        room = room_name.lower().strip()

        # Define relevance mapping: {object: {room: relevance_score}}
        relevance_map = {
            'toilet': {'bathroom': 1.0, 'unknown': 0.3, 'bedroom': 0.1},
            'bathtub': {'bathroom': 1.0, 'unknown': 0.3},
            'shower': {'bathroom': 1.0, 'gym': 0.3, 'unknown': 0.3},
            'sink': {'bathroom': 0.9, 'kitchen': 0.9, 'laundry room': 0.5, 'unknown': 0.3},
            'bed': {'bedroom': 1.0, 'unknown': 0.2},
            'sofa': {'living room': 1.0, 'lounge': 0.9, 'office room': 0.3, 'unknown': 0.2},
            'table': {'dining room': 0.9, 'kitchen': 0.8, 'living room': 0.6, 'office room': 0.7, 'unknown': 0.3},
            'chair': {'dining room': 0.8, 'kitchen': 0.7, 'living room': 0.6, 'office room': 0.9, 'bedroom': 0.4, 'unknown': 0.3},
            'stove': {'kitchen': 1.0, 'unknown': 0.2},
            'refrigerator': {'kitchen': 1.0, 'unknown': 0.2},
            'counter': {'kitchen': 0.9, 'bathroom': 0.6, 'unknown': 0.3},
            'tv': {'living room': 0.9, 'bedroom': 0.7, 'lounge': 0.8, 'unknown': 0.2},
            'treadmill': {'gym': 1.0, 'bedroom': 0.3, 'unknown': 0.2},
            'gym_equipment': {'gym': 1.0, 'unknown': 0.2},
            'washer': {'laundry room': 1.0, 'bathroom': 0.3, 'unknown': 0.2},
            'dryer': {'laundry room': 1.0, 'unknown': 0.2},
        }

        # Try to find relevance in the map
        if goal in relevance_map:
            room_scores = relevance_map[goal]
            if room in room_scores:
                return room_scores[room]
            # Default relevance for rooms not in the map
            return 0.1

        # Default: medium relevance for unknown goal-room pairs
        return 0.3

    def _get_exploration_context(self, goal_description):
        """
        Generate exploration context description for landmark comparison.

        Provides background information about:
        - Overall exploration progress (in square meters)
        - Detected rooms and their exploration status
        - Relevance of rooms to the goal

        Args:
            goal_description: The goal object being searched for

        Returns:
            str: Formatted exploration context string
        """
        context_lines = []

        # 1. Overall exploration progress (in square meters)
        overall_stats = self._get_overall_exploration_stats()
        if overall_stats:
            explored_area_sqm = overall_stats['explored_area_sqm']

            # Categorize exploration by area explored
            if explored_area_sqm < 20:
                status = "Early exploration"
            elif explored_area_sqm < 60:
                status = "Moderate exploration"
            elif explored_area_sqm < 120:
                status = "Advanced exploration"
            else:
                status = "Extensive exploration"

            context_lines.append(f"- Overall Progress: {status} ({explored_area_sqm:.1f} m² mapped)")

        # 2. Room coverage analysis
        room_info = self._analyze_room_coverage()

        if len(room_info) > 0:
            # Categorize rooms by exploration status
            thoroughly_explored = [r for r in room_info if r['coverage'] > 0.85]
            partially_explored = [r for r in room_info if 0.3 < r['coverage'] <= 0.85]
            barely_explored = [r for r in room_info if r['coverage'] <= 0.3]

            room_status_parts = []

            if len(thoroughly_explored) > 0:
                rooms_str = ', '.join([r['name'] for r in thoroughly_explored])
                room_status_parts.append(f"Thoroughly explored (>85%): {rooms_str}")

            if len(partially_explored) > 0:
                rooms_with_pct = ', '.join([f"{r['name']} ({r['coverage']*100:.0f}%)" for r in partially_explored])
                room_status_parts.append(f"Partially explored: {rooms_with_pct}")

            if len(barely_explored) > 0:
                rooms_str = ', '.join([r['name'] for r in barely_explored])
                room_status_parts.append(f"Barely explored (<30%): {rooms_str}")

            if room_status_parts:
                context_lines.append("- Room Status:")
                for part in room_status_parts:
                    context_lines.append(f"  • {part}")

            # 3. Goal relevance analysis
            # Sort rooms by relevance to goal
            for r in room_info:
                r['relevance'] = self._compute_goal_room_relevance(goal_description, r['name'])

            high_relevance_rooms = [r for r in room_info if r['relevance'] >= 0.7]
            medium_relevance_rooms = [r for r in room_info if 0.3 <= r['relevance'] < 0.7]

            if high_relevance_rooms or medium_relevance_rooms:
                context_lines.append(f"- Goal Relevance for '{goal_description}':")

                for r in high_relevance_rooms:
                    if r['coverage'] < 0.5:
                        status = "barely explored - strong candidate for investigation"
                    elif r['coverage'] < 0.85:
                        status = "partially explored - may need further search"
                    else:
                        status = "thoroughly explored - likely already checked"
                    context_lines.append(f"  • {r['name']} (high relevance, {status})")

                for r in medium_relevance_rooms:
                    context_lines.append(f"  • {r['name']} (medium relevance, {r['coverage']*100:.0f}% explored)")

                # Note about potentially undiscovered rooms
                if overall_stats and explored_area_sqm < 100:
                    context_lines.append(f"  • Additional rooms may exist in unexplored areas")
        else:
            context_lines.append("- Room Status: No rooms confidently detected yet")
            if overall_stats:
                explored_area_sqm = overall_stats['explored_area_sqm']
                if explored_area_sqm < 40:
                    context_lines.append(f"  • Still in early exploration phase ({explored_area_sqm:.1f} m² mapped)")

        if len(context_lines) == 0:
            return ""

        return "Exploration Context:\n" + "\n".join(context_lines) + "\n"
