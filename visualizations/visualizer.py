import copy
import os
from matplotlib import colors
import cv2
import numpy as np
import skimage

from utils.image_process import (
    add_resized_image,
    add_rectangle,
    add_text,
    add_text_list,
    draw_agent,
    draw_goal,
    line_list,
    draw_landmark,
    draw_bounding_boxes_with_captions,
    draw_landmarks_on_rgb_image,
)

from .rag_panel import create_rag_panel, create_rag_compact_info

# Import visualization configuration
from configs.detector_config import USE_DINO_VISUALIZATION


def create_annotated_observation(agent, use_dino=None):
    """Create an annotated observation image with bounding boxes and captions.

    Automatically follows the detection path used by navigation logic:
    - If goal is a small object: shows GroundingDINO scene graph results
    - If goal is a large object: shows navigation detector (FastSAM+CLIP or GLIP) results

    Args:
        agent: The agent instance containing detection results
        use_dino: If True, force use GroundingDINO; if False, force use navigation detector.
                  If None (default), automatically follows agent.active_detection_path.

    Returns:
        Annotated RGB image with bounding boxes and captions
    """
    # Start with the original observation
    base_image = agent.rgb_visualization.copy()

    # Determine which detection results to use
    if use_dino is None:
        # Automatically follow the detection path used by navigation logic
        if hasattr(agent, 'active_detection_path'):
            use_dino = (agent.active_detection_path == 'scenegraph')
        else:
            # Fallback to config if agent doesn't have active_detection_path
            use_dino = USE_DINO_VISUALIZATION

    if use_dino:
        # Use GroundingDINO scene graph results
        return _create_annotated_from_scenegraph(agent, base_image)
    else:
        # Use navigation detector results (GLIP or FastSAM)
        return _create_annotated_from_navigation(agent, base_image)


def _create_annotated_from_navigation(agent, base_image):
    """Create annotation using navigation detector results (only goal objects)."""
    # Check if we have navigation detection results
    if not hasattr(agent, 'current_obj_predictions'):
        return base_image

    predictions = agent.current_obj_predictions

    # Check if predictions is empty, None, or is an empty list (initial state)
    if predictions is None or predictions == [] or (hasattr(predictions, 'bbox') and len(predictions.bbox) == 0):
        return base_image

    # Get goal object name
    goal_name = agent.obj_goal if hasattr(agent, 'obj_goal') else None
    if goal_name is None:
        return base_image

    # Extract bounding boxes, labels, and scores from navigation detector
    try:
        # Get bboxes (convert to numpy if tensor)
        if hasattr(predictions.bbox, 'cpu'):
            all_bboxes = predictions.bbox.cpu().numpy()
        else:
            all_bboxes = predictions.bbox

        # Get labels
        labels = predictions.get_field("labels")
        if hasattr(labels, 'tolist'):
            all_labels = [str(label) for label in labels.tolist()]
        elif isinstance(labels, list):
            all_labels = [str(label) for label in labels]
        else:
            all_labels = [str(label) for label in labels]

        # Get scores
        scores = predictions.get_field("scores")
        if hasattr(scores, 'cpu'):
            all_scores = scores.cpu().numpy()
        else:
            all_scores = scores

    except Exception as e:
        # If there's any error (e.g., predictions is a list), return base image
        return base_image

    # Filter: only keep goal objects
    filtered_bboxes = []
    filtered_captions = []
    filtered_scores = []

    for i, label in enumerate(all_labels):
        if label == goal_name:
            filtered_bboxes.append(all_bboxes[i])
            filtered_captions.append(label)
            filtered_scores.append(all_scores[i])

    if len(filtered_bboxes) == 0:
        return base_image

    # Draw bounding boxes with captions for goal objects only
    annotated_image = draw_bounding_boxes_with_captions(
        base_image,
        filtered_bboxes,
        filtered_captions,
        filtered_scores
    )

    return annotated_image


def _create_annotated_from_scenegraph(agent, base_image):
    """Create annotation using GroundingDINO scene graph results (only goal objects)."""
    # Check if we have segment2d results
    if not hasattr(agent, 'scenegraph') or not hasattr(agent.scenegraph, 'segment2d_results'):
        return base_image

    segment2d_results = agent.scenegraph.segment2d_results

    if len(segment2d_results) == 0:
        return base_image

    # Get goal object name (use scene graph version with underscore)
    goal_name_sg = agent.obj_goal_sg if hasattr(agent, 'obj_goal_sg') else None
    if goal_name_sg is None:
        return base_image

    # Get the most recent segmentation result
    latest_result = segment2d_results[-1]

    # Extract bounding boxes and captions
    if 'xyxy' not in latest_result or 'caption' not in latest_result:
        return base_image

    all_bboxes = latest_result['xyxy']
    all_captions = latest_result['caption']
    all_confidences = latest_result.get('confidence', None)

    if len(all_bboxes) == 0 or len(all_captions) == 0:
        return base_image

    # Filter: only keep goal objects (check if goal_name_sg is in caption)
    filtered_bboxes = []
    filtered_captions = []
    filtered_scores = []

    for i, caption in enumerate(all_captions):
        # GroundingDINO caption might be like "cabinet drawers", we check if goal is in the caption
        if goal_name_sg in caption.split():
            filtered_bboxes.append(all_bboxes[i])
            filtered_captions.append(caption)
            if all_confidences is not None:
                filtered_scores.append(all_confidences[i])
            else:
                filtered_scores.append(0.9)  # Default confidence

    if len(filtered_bboxes) == 0:
        return base_image

    # Draw bounding boxes with captions for goal objects only
    annotated_image = draw_bounding_boxes_with_captions(
        base_image,
        filtered_bboxes,
        filtered_captions,
        filtered_scores if len(filtered_scores) > 0 else None
    )

    return annotated_image


def format_scenegraph_combinations(agent, max_line_length=70, max_lines=5):
    """Format the scene graph combination with detailed statistics.

    Line 1: Node statistics (object nodes, group nodes, room nodes with objects)
    Line 2: Best scene graph structure with grouping notation
    Line 3: Objects per room
    Line 4: Objects per group

    Args:
        agent: Agent instance with scenegraph
        max_line_length: Maximum characters per line
        max_lines: Maximum number of lines to display

    Returns:
        str: Formatted text describing the scene graph
    """
    lines = []

    # Line 1: Node statistics
    num_objects = len(agent.scenegraph.nodes) if hasattr(agent, 'scenegraph') and hasattr(agent.scenegraph, 'nodes') else 0

    # Count groups and rooms with objects
    num_groups = 0
    num_rooms_with_objects = 0
    if hasattr(agent, 'scenegraph') and hasattr(agent.scenegraph, 'room_nodes'):
        for room_node in agent.scenegraph.room_nodes:
            if len(room_node.nodes) > 0:
                num_rooms_with_objects += 1
                num_groups += len(room_node.group_nodes)

    lines.append(f"Nodes: {num_objects} objs, {num_groups} groups, {num_rooms_with_objects} rooms")

    # Get best scene graph combination
    # If scenegraph_combinations doesn't exist or is empty, fallback to actual room_nodes
    use_actual_rooms = False
    if not hasattr(agent, 'scenegraph_combinations') or agent.scenegraph_combinations is None:
        use_actual_rooms = True
    else:
        combos = agent.scenegraph_combinations.get_top_k(1)
        if not combos:
            use_actual_rooms = True
        else:
            combo = combos[0]
            prob = combo.get('probability', 0.0)
            room_combos = combo.get('room_combinations', [])

    # Fallback: use actual room_nodes from scenegraph
    if use_actual_rooms:
        lines.append("SG: Using actual room assignments")

        # Build room_combos from actual room_nodes
        room_combos = []
        if hasattr(agent, 'scenegraph') and hasattr(agent.scenegraph, 'room_nodes'):
            for room_node in agent.scenegraph.room_nodes:
                # Only include rooms that have objects
                if len(room_node.nodes) == 0:
                    continue

                room_combo = {
                    'room_caption': room_node.caption,
                    'group_nodes': room_node.group_nodes if hasattr(room_node, 'group_nodes') else []
                }
                room_combos.append(room_combo)

        if not room_combos:
            lines.append("No rooms with objects")
            return "\n".join(lines)

        prob = 1.0  # Actual data has 100% probability

    # Line 2: Scene graph structure with notation
    # Format: [room1: (g1: obj1, obj2) (g2: obj3)] [room2: (g3: obj4)]
    sg_structure = f"SG({prob*100:.0f}%): "
    for room_combo in room_combos:
        room_name = room_combo.get('room_caption', 'unknown')
        group_nodes = room_combo.get('group_nodes', [])

        if not group_nodes:
            continue

        # Start room bracket
        room_str = f"[{room_name}: "

        # Add each group with parentheses
        group_strs = []
        for gn in group_nodes:
            obj_captions = []
            if hasattr(gn, 'nodes') and gn.nodes:
                for obj_node in gn.nodes:
                    if hasattr(obj_node, 'caption') and obj_node.caption:
                        obj_captions.append(obj_node.caption)

            if obj_captions:
                group_strs.append(f"({', '.join(obj_captions[:2])})")  # Show max 2 objects per group

        room_str += ' '.join(group_strs) + "]"
        sg_structure += room_str + " "

    # Truncate if too long
    if len(sg_structure) > max_line_length:
        sg_structure = sg_structure[:max_line_length-3] + "..."
    lines.append(sg_structure)

    # Line 3: Objects per room
    room_obj_summary = "Rooms: "
    for room_combo in room_combos:
        room_name = room_combo.get('room_caption', 'unknown')
        group_nodes = room_combo.get('group_nodes', [])

        # Count total objects in this room
        total_objs = sum(len(gn.nodes) for gn in group_nodes if hasattr(gn, 'nodes'))
        room_obj_summary += f"{room_name}({total_objs}objs) "

    if len(room_obj_summary) > max_line_length:
        room_obj_summary = room_obj_summary[:max_line_length-3] + "..."
    lines.append(room_obj_summary)

    # Line 4: Objects per group
    group_obj_summary = "Groups: "
    group_id = 1
    for room_combo in room_combos:
        group_nodes = room_combo.get('group_nodes', [])
        for gn in group_nodes:
            num_objs = len(gn.nodes) if hasattr(gn, 'nodes') else 0
            group_obj_summary += f"G{group_id}({num_objs}) "
            group_id += 1

    if len(group_obj_summary) > max_line_length:
        group_obj_summary = group_obj_summary[:max_line_length-3] + "..."
    lines.append(group_obj_summary)

    return "\n".join(lines[:max_lines])



def create_room_map_visualization(agent, agent_coordinate, crop_size=(150, 200)):
    """Create a visualization of the room_map showing different room types in different colors.

    Args:
        agent: The agent instance containing room_map
        agent_coordinate: (x, y) position of the agent (kept for compatibility, not used for cropping)
        crop_size: Size parameter (kept for compatibility, not used - full map is returned)

    Returns:
        RGB image of the full room map visualization
    """
    import torch

    # Room type colors (BGR format for OpenCV)
    # Note: room_map has 9 channels (bedroom-laundry), but room_nodes has 10 (unknown + 9 detected)
    # We add 1 to room indices when visualizing detected rooms
    room_colors = {
        0: (220, 220, 220),   # unknown - light gray (BGR)
        1: (193, 182, 255),   # bedroom - light pink (BGR)
        2: (144, 238, 144),   # living room - light green (BGR)
        3: (230, 216, 173),   # bathroom - light blue (BGR)
        4: (185, 218, 255),   # kitchen - peach (BGR)
        5: (221, 160, 221),   # dining room - plum (BGR)
        6: (140, 230, 240),   # office room - khaki (BGR)
        7: (122, 160, 255),   # gym - light salmon (BGR)
        8: (230, 224, 176),   # lounge - powder blue (BGR)
        9: (211, 211, 211),   # laundry room - darker gray (BGR)
    }

    if not hasattr(agent, 'room_map') or agent.room_map is None:
        # Return a blank image if no room_map
        blank = np.full((crop_size[1], crop_size[0], 3), 255, dtype=np.uint8)
        return blank

    room_map = agent.room_map.cpu().numpy()[0]  # [9, H, W]
    H, W = room_map.shape[1], room_map.shape[2]

    # Create RGB visualization
    room_vis = np.full((H, W, 3), 255, dtype=np.uint8)  # White background

    # Get the dominant room type at each position
    room_sum = room_map.sum(axis=0)  # Sum across all room channels
    has_room_info = room_sum > 0  # Positions with any room information

    # For positions with room info, get the argmax
    if has_room_info.any():
        dominant_room = room_map.argmax(axis=0)  # [H, W], values in [0-8]
        # Add 1 to match room_nodes indexing (0=unknown, 1-9=detected rooms)
        dominant_room = dominant_room + 1

        # Color each position by its dominant room type
        for room_idx, color in room_colors.items():
            if room_idx == 0:
                continue  # Skip unknown, only color detected rooms
            mask = (dominant_room == room_idx) & has_room_info
            # Flip vertically to match map orientation
            room_vis[::-1][mask] = color

    # Draw object nodes on the map
    if hasattr(agent, 'scenegraph') and hasattr(agent.scenegraph, 'nodes'):
        for obj_node in agent.scenegraph.nodes:
            if hasattr(obj_node, 'center') and obj_node.center is not None:
                obj_x, obj_y = obj_node.center
                # obj_y is already flipped in the scenegraph
                if 0 <= int(obj_x) < W and 0 <= int(obj_y) < H:
                    # Draw small circles for objects
                    if hasattr(obj_node, 'is_goal_node') and obj_node.is_goal_node:
                        # Goal objects in green
                        cv2.circle(room_vis, (int(obj_x), int(obj_y)), 2, (0, 255, 0), -1)
                    else:
                        # Regular objects in black
                        cv2.circle(room_vis, (int(obj_x), int(obj_y)), 2, (0, 0, 0), -1)
                    # Add white border for visibility
                    cv2.circle(room_vis, (int(obj_x), int(obj_y)), 2, (255, 255, 255), 1)

    # Draw landmarks and edges on the room map
    room_vis = draw_landmarks_on_rgb_image(room_vis, agent, edge_color=(0, 255, 255), edge_thickness=1)

    return room_vis


def visualize(agent, traversible):
    """Extracted visualization function. Accepts the agent instance and the
    traversible map. Mirrors the original behavior from `PSG_Nav.py` but lives
    in this module to keep `PSG_Nav.py` focused on agent logic.

    Layout (high-resolution, 2x scale):
    - Top row: Observation (640x480) | Occupancy Map + Agent History (440x480) | Room Map (440x480)
    - Middle row: Scene Graph info | Room Legend
    - Bottom row: Navigation Status (with RAG compact info)
    - Right panel: RAG Panel (500x960)

    Agent History is integrated into Occupancy Map as semi-transparent green overlay:
    - Light green (15% opacity): Expanded visited area (~0.5m radius)
    - Medium green (35% opacity): Core visited path

    Canvas size: 1100 (height) x 2140 (width) - 2x higher resolution
    """
    import torch
    from utils.utils_frontiers import calculate_frontiers

    save_map = copy.deepcopy(torch.from_numpy(traversible))
    gray_map = torch.stack((save_map, save_map, save_map))
    paper_obstacle_map = copy.deepcopy(gray_map)[:, 1:-1, 1:-1]
    paper_map = torch.zeros_like(paper_obstacle_map)
    paper_map_trans = paper_map.permute(1, 2, 0)
    unknown_rgb = colors.to_rgb('#FFFFFF')
    paper_map_trans[:, :, :] = torch.tensor(unknown_rgb)
    free_rgb = colors.to_rgb('#E7E7E7')
    paper_map_trans[agent.fbe_free_map.cpu().numpy()[0, 0, ::-1] > 0.5, :] = torch.tensor(free_rgb).double()
    obstacle_rgb = colors.to_rgb('#A2A2A2')
    paper_map_trans[skimage.morphology.binary_dilation(agent.full_map.cpu().numpy()[0, 0, ::-1] > 0.5, skimage.morphology.disk(1)), :] = torch.tensor(obstacle_rgb).double()
    paper_map_trans = paper_map_trans.permute(2, 0, 1)

    # Color obstacles by caption from scene graph nodes
    # dilation_size controls how much to expand each object's colored region
    # draw_objects_by_caption(agent, paper_map_trans, dilation_size=1)

    # Calculate frontiers for visualization
    frontier_map, frontier_locations, num_frontiers = calculate_frontiers(agent.full_map, agent.fbe_free_map)

    visualize_agent_and_goal(agent, paper_map_trans, frontier_locations)
    agent_coordinate = (
        int(agent.history_pose[-1][0] * 100 / agent.resolution),
        int((agent.map_size_cm / 100 - agent.history_pose[-1][1]) * 100 / agent.resolution),
    )
    # Convert to RGB numpy array for drawing
    occupancy_map_full = (paper_map_trans.permute(1, 2, 0) * 255).numpy().astype(np.uint8)

    # Draw landmarks on the full map before cropping
    occupancy_map_full = draw_landmarks_on_rgb_image(occupancy_map_full, agent, edge_color=(0, 255, 255), edge_thickness=3)

    # Save full occupancy map before cropping
    # Convert RGB to BGR for cv2.imwrite
    occupancy_map_full_bgr = occupancy_map_full[:, :, ::-1]
    occupancy_maps_dir = os.path.join(agent.visualization_dir, 'occupancy_maps', f'episode_{agent.count_episodes:06d}')
    os.makedirs(occupancy_maps_dir, exist_ok=True)
    frame_number = len(agent.visualize_image_list)
    occupancy_map_path = os.path.join(occupancy_maps_dir, f'occupancy_map_{frame_number:06d}.png')
    cv2.imwrite(occupancy_map_path, occupancy_map_full_bgr)

    # Use full occupancy map instead of cropping
    occupancy_map = occupancy_map_full

    # Create room map visualization - larger crop size (200x200)
    room_map_vis = create_room_map_visualization(agent, agent_coordinate, crop_size=(250, 250))

    # High-resolution canvas: 2x scale (1100 x 2140)
    visualize_image = np.full((1100, 2140, 3), 255, dtype=np.uint8)

    # Create annotated observation with bounding boxes and captions
    annotated_observation = create_annotated_observation(agent)
    visualize_image = add_resized_image(visualize_image, annotated_observation, (20, 120), (640, 480))
    # Both maps displayed at size (440x480)
    visualize_image = add_resized_image(visualize_image, occupancy_map, (680, 120), (440, 480))
    visualize_image = add_resized_image(visualize_image, room_map_vis, (1140, 120), (440, 480))

    # Rectangles for each section (2x scale, thickness=2) - maps are now 440x480
    visualize_image = add_rectangle(visualize_image, (20, 120), (660, 600), (128, 128, 128), thickness=2)    # Observation
    visualize_image = add_rectangle(visualize_image, (680, 120), (1120, 600), (128, 128, 128), thickness=2)   # Occupancy Map (440x480)
    visualize_image = add_rectangle(visualize_image, (1140, 120), (1580, 600), (128, 128, 128), thickness=2)   # Room Map (440x480)
    visualize_image = add_rectangle(visualize_image, (20, 620), (1040, 800), (128, 128, 128), thickness=2)   # SceneGraph
    visualize_image = add_rectangle(visualize_image, (1060, 620), (1580, 800), (128, 128, 128), thickness=2)  # Room Legend
    visualize_image = add_rectangle(visualize_image, (20, 900), (1580, 1080), (128, 128, 128), thickness=2)   # Navigation Status

    # Add RAG panel on the right side (2x scale)
    if hasattr(agent, 'rag_manager') or hasattr(agent, 'rag_best_detection'):
        try:
            rag_panel = create_rag_panel(agent, panel_size=(500, 960))
            visualize_image = add_resized_image(visualize_image, rag_panel, (1620, 120), (500, 960))
            visualize_image = add_rectangle(visualize_image, (1620, 120), (2120, 1080), (128, 128, 128), thickness=2)
        except Exception as e:
            # Fallback: just show error text
            visualize_image = add_text(visualize_image, "RAG: Error", (820, 80), font_scale=0.4, thickness=1)
            print(f"RAG panel creation error: {e}")

    # Section titles
    visualize_image = add_text(visualize_image, "Observation (Goal: {})".format(agent.obj_goal), (140, 100), font_scale=1.0, thickness=2)
    visualize_image = add_text(visualize_image, "Occupancy Map", (840, 100), font_scale=1.0, thickness=2)
    visualize_image = add_text(visualize_image, "Room Map", (1360, 100), font_scale=1.0, thickness=2)
    visualize_image = add_text(visualize_image, "Scene Graph", (400, 610), font_scale=1.0, thickness=2)
    visualize_image = add_text(visualize_image, "Room Legend", (1240, 610), font_scale=1.0, thickness=2)
    visualize_image = add_text(visualize_image, "Navigation Status", (700, 880), font_scale=1.0, thickness=2)

    # Format and display the best scene graph combination (2x scale)
    sg_combo_text = format_scenegraph_combinations(agent, max_line_length=70, max_lines=5)
    visualize_image = add_text_list(visualize_image, sg_combo_text.split('\n'), (40, 660), font_scale=0.9, thickness=2)

    # Room legend with colors (from remote/HEAD) - excluding unknown
    room_names = ['bedroom', 'living room', 'bathroom', 'kitchen', 'dining room',
                  'office room', 'gym', 'lounge', 'laundry room']
    room_colors_bgr = [
        (193, 182, 255),   # bedroom - light pink
        (144, 238, 144),   # living room - light green
        (230, 216, 173),   # bathroom - light blue
        (185, 218, 255),   # kitchen - peach
        (221, 160, 221),   # dining room - plum
        (140, 230, 240),   # office room - khaki
        (122, 160, 255),   # gym - light salmon
        (230, 224, 176),   # lounge - powder blue
        (211, 211, 211),   # laundry room - darker gray
    ]
    legend_y = 650
    for i, (name, color) in enumerate(zip(room_names, room_colors_bgr)):
        col = i % 3
        row = i // 3
        x = 1080 + col * 170
        y = legend_y + row * 44
        # Draw color square (2x scale)
        cv2.rectangle(visualize_image, (x, y), (x + 24, y + 24), color, -1)
        cv2.rectangle(visualize_image, (x, y), (x + 24, y + 24), (128, 128, 128), 2)
        # Draw room name (2x scale)
        visualize_image = add_text(visualize_image, name[:8], (x + 32, y + 22), font_scale=0.9, thickness=2)

    # Navigation status at bottom (2x scale)
    nav_explanation = agent.explanation

    # Add agent position and orientation
    if hasattr(agent, 'history_pose') and len(agent.history_pose) > 0:
        current_pose = agent.history_pose[-1]  # [x, y, orientation]
        agent_orientation = current_pose[2] if len(current_pose) > 2 else 0.0

        # Add position and orientation info
        nav_explanation = f"Agent Position: ({agent.agent_map_pos}) | Orientation: {agent_orientation:.1f} degree.  {nav_explanation}"

    # Add distance to goal if available
    if hasattr(agent, 'distance_to_goal') and agent.distance_to_goal is not None:
        nav_explanation = f"{nav_explanation}; Distance to Goal: {agent.distance_to_goal:.2f}m. "

    if hasattr(agent, 'rag_manager') or hasattr(agent, 'rag_best_detection'):
        rag_info = create_rag_compact_info(agent)
        nav_explanation = f"{nav_explanation}\n{rag_info}"

    visualize_image = add_text_list(visualize_image, line_list(nav_explanation, 160), (40, 940), font_scale=0.9, thickness=2)

    visualize_image = visualize_image[:, :, ::-1]

    # Save this frame as an image (commented out - not needed)
    frames_dir = os.path.join(agent.visualization_dir, 'frames', f'episode_{agent.count_episodes:06d}')
    os.makedirs(frames_dir, exist_ok=True)
    frame_number = len(agent.visualize_image_list)
    frame_path = os.path.join(frames_dir, f'frame_{frame_number:06d}.png')
    cv2.imwrite(frame_path, visualize_image)

    agent.visualize_image_list.append(visualize_image)


def save_video(agent):
    """Save the accumulated frames for the agent into a video file (tries ffmpeg
    first, falls back to OpenCV codecs, and finally saves individual frames).
    """
    save_video_dir = os.path.join(agent.visualization_dir, 'video')
    save_video_path = f'{save_video_dir}/vid_{agent.count_episodes:06d}.mp4'
    if not os.path.exists(save_video_dir):
        os.makedirs(save_video_dir)

    if len(agent.visualize_image_list) == 0:
        print("Warning: No images to save in video")
        return

    height, width, _ = agent.visualize_image_list[0].shape

    # Method 1: Try FFmpeg first (most compatible MP4)
    try:
        import subprocess
        frames_dir_temp = os.path.join(save_video_dir, f'temp_frames_{agent.count_episodes:06d}')
        os.makedirs(frames_dir_temp, exist_ok=True)

        print("Using FFmpeg for maximum MP4 compatibility...")
        # Save frames as temporary PNG files
        for i, frame in enumerate(agent.visualize_image_list):
            frame_path = os.path.join(frames_dir_temp, f'frame_{i:06d}.png')
            cv2.imwrite(frame_path, frame)

        # Use FFmpeg to create MP4
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '4',
            '-i', os.path.join(frames_dir_temp, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-movflags', '+faststart',
            save_video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0 and os.path.exists(save_video_path):
            file_size = os.path.getsize(save_video_path)
            print(f"✅ FFmpeg MP4 saved successfully: {save_video_path}")
            print(f"- File size: {file_size} bytes")

            # Verify the file can be read
            test_cap = cv2.VideoCapture(save_video_path)
            if test_cap.isOpened():
                test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_cap.release()
                print(f"- Verified: {test_frame_count} frames readable")

                # Clean up temporary frames
                import shutil
                shutil.rmtree(frames_dir_temp)
                return

    except Exception as e:
        print(f"FFmpeg method failed: {e}")
        # Clean up temp frames if they exist
        try:
            import shutil
            if os.path.exists(frames_dir_temp):
                shutil.rmtree(frames_dir_temp)
        except:
            pass

    # Method 2: Fallback to OpenCV with improved MP4 support

    # Try different codecs in order of preference
    codecs_to_try = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]

    for codec, ext in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_path_with_ext = save_video_path.replace('.mp4', ext)

            if ext == '.mp4':
                video = cv2.VideoWriter(video_path_with_ext, fourcc, 4.0, (width, height), True)
            else:
                video = cv2.VideoWriter(video_path_with_ext, fourcc, 4.0, (width, height))

            if not video.isOpened():
                continue

            print(f"Using codec {codec}, saving to {video_path_with_ext}")

            frames_written = 0
            for i, visualize_image in enumerate(agent.visualize_image_list):
                if visualize_image.dtype != np.uint8:
                    visualize_image = visualize_image.astype(np.uint8)
                if visualize_image.shape[:2] != (height, width):
                    visualize_image = cv2.resize(visualize_image, (width, height))
                if len(visualize_image.shape) == 3 and visualize_image.shape[2] == 3:
                    video.write(visualize_image)
                    frames_written += 1
                else:
                    print(f"Warning: Frame {i} has unexpected shape {visualize_image.shape}")

            video.release()

            if os.path.exists(video_path_with_ext) and os.path.getsize(video_path_with_ext) > 1000:
                print(f"Video saved successfully: {video_path_with_ext}")
                print(f"- Frames written: {frames_written}")
                print(f"- File size: {os.path.getsize(video_path_with_ext)} bytes")

                test_cap = cv2.VideoCapture(video_path_with_ext)
                if test_cap.isOpened():
                    test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    test_cap.release()
                    print(f"- Verified: {test_frame_count} frames readable")
                    return
                else:
                    print(f"Warning: Saved file cannot be reopened")
            else:
                print(f"Warning: Saved file is invalid or too small")

        except Exception as e:
            print(f"Failed to save with codec {codec}: {e}")
            continue

    # Final fallback: save individual frames
    print("All video encoding methods failed. Saving individual frames...")
    frames_dir = os.path.join(save_video_dir, f'frames_{agent.count_episodes:06d}')
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(agent.visualize_image_list):
        frame_path = os.path.join(frames_dir, f'frame_{i:06d}.png')
        cv2.imwrite(frame_path, frame)
    print(f"Frames saved to: {frames_dir}")


def visualize_agent_and_goal(agent, map, frontier_locations=None):
    for idx, pose in enumerate(agent.history_pose):
        draw_step_num = 50  # Increased for smoother gradient
        min_alpha = 0.5  # Minimum alpha to keep all trajectories visible
        # Calculate alpha with minimum threshold
        alpha_raw = 1 - (len(agent.history_pose) - idx) / draw_step_num
        alpha = max(min_alpha, alpha_raw)  # Ensure all trajectories are at least 50% visible
        agent_size = 3  # Increased from 1 for better visibility
        if idx == len(agent.history_pose) - 1:
            agent_size = 5  # Increased from 2 for better visibility
        draw_agent(agent=agent, map=map, pose=pose, agent_size=agent_size, color_index=0, alpha=alpha)
    draw_goal(agent=agent, map=map, goal_size=8, color_index=1)  # Increased from 4 for better visibility
    # Draw landmarks (blue) only, without frontiers (yellow)
    draw_landmark(agent=agent, map=map, landmark_size=5, color_index=2,  # Increased from 2 for better visibility
                  frontier_locations=None, frontier_color=(1.0, 1.0, 0.0))

    return map
