import cv2
import numpy as np
import torch
import skimage
from PIL import Image, ImageDraw, ImageFont

# color channel indices used in map (map shape assumed (C, H, W))
# adjust these if your map channels are arranged differently
COLOR_AGENT = 0
COLOR_GOAL = 1
COLOR_LANDMARK = 2

def line_list(text, line_length=80):
    text_list = []
    for i in range(0, len(text), line_length):
        text_list.append(text[i:(i + line_length)])
    return text_list

def add_text(image: np.ndarray, text: str, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    """Add text to image with Unicode support using PIL.

    Args:
        image: numpy array image (BGR format from OpenCV)
        text: text string (supports Unicode characters like °)
        position: (x, y) tuple for text position
        font: ignored (kept for compatibility)
        font_scale: scale factor for font size
        color: BGR color tuple (OpenCV format)
        thickness: ignored (kept for compatibility)
    """
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Calculate font size based on font_scale (approximation)
    font_size = int(20 * font_scale)

    # Try to load a TrueType font, fall back to default if not available
    try:
        # Try to use DejaVu Sans which supports many Unicode characters
        pil_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            # Try alternative font paths
            pil_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
        except:
            # Fall back to default font
            pil_font = ImageFont.load_default()

    # Convert BGR color to RGB for PIL
    rgb_color = (color[2], color[1], color[0])

    # Draw text
    draw.text(position, text, font=pil_font, fill=rgb_color)

    # Convert back to BGR for OpenCV
    image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Copy the result back to the original image
    image[:] = image_bgr
    return image

def add_text_list(image: np.ndarray, text_list: list, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    """Add multiple lines of text to image with Unicode support using PIL.

    Args:
        image: numpy array image (BGR format from OpenCV)
        text_list: list of text strings (supports Unicode characters like °)
        position: (x, y) tuple for first line position
        font: ignored (kept for compatibility)
        font_scale: scale factor for font size
        color: BGR color tuple (OpenCV format)
        thickness: ignored (kept for compatibility)
    """
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Calculate font size based on font_scale
    font_size = int(20 * font_scale)

    # Try to load a TrueType font
    try:
        pil_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            pil_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
        except:
            pil_font = ImageFont.load_default()

    # Convert BGR color to RGB for PIL
    rgb_color = (color[2], color[1], color[0])

    # Calculate line height based on font size
    line_height = int(font_size * 1.5)

    # Draw each line of text
    for i, text in enumerate(text_list):
        position_i = (position[0], position[1] + i * line_height)
        draw.text(position_i, text, font=pil_font, fill=rgb_color)

    # Convert back to BGR for OpenCV
    image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Copy the result back to the original image
    image[:] = image_bgr
    return image

def add_rectangle(image: np.ndarray, top_left: tuple, bottom_right: tuple, color=(0, 255, 0), thickness=2):
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image

def add_resized_image(base_image: np.ndarray, overlay_image: np.ndarray, position: tuple, size: tuple):
    resized_overlay = cv2.resize(overlay_image, size)

    h, w = resized_overlay.shape[:2]

    x, y = position

    if x + w > base_image.shape[1] or y + h > base_image.shape[0]:
        raise ValueError("Overlay image goes out of the bounds of the base image.")

    base_image[y:y+h, x:x+w] = resized_overlay
    return base_image

def crop_around_point(image: np.ndarray, point: tuple, size: tuple):
    img_height, img_width = image.shape[:2]
    
    crop_width, crop_height = size
    
    left = max(point[0] - crop_width // 2, 0)
    top = max(point[1] - crop_height // 2, 0)
    right = min(point[0] + (crop_width - crop_width // 2), img_width)
    bottom = min(point[1] + (crop_height - crop_height // 2), img_height)
    
    if right - left < crop_width:
        if left == 0:
            right = left + crop_width
        else:
            left = right - crop_width
    if bottom - top < crop_height:
        if top == 0:
            bottom = top + crop_height
        else:
            top = bottom - crop_height
    
    cropped_image = image[top:bottom, left:right]
    
    return cropped_image

def draw_agent(agent, map, pose, agent_size, color_index, alpha=1):
    color_ori = map[:, int((agent.map_size_cm/100-pose[1])*100/agent.resolution)-agent_size:int((agent.map_size_cm/100-pose[1])*100/agent.resolution)+agent_size, int(pose[0]*100/agent.resolution)-agent_size:int(pose[0]*100/agent.resolution)+agent_size]
    color_new = torch.zeros_like(color_ori)
    color_new[color_index] = 1
    color_new = alpha * color_new + (1 - alpha) * color_ori
    map[:, int((agent.map_size_cm/100-pose[1])*100/agent.resolution)-agent_size:int((agent.map_size_cm/100-pose[1])*100/agent.resolution)+agent_size, int(pose[0]*100/agent.resolution)-agent_size:int(pose[0]*100/agent.resolution)+agent_size] = color_new

def draw_goal(agent, map, goal_size, color_index):
    skimage.morphology.disk(goal_size)

    # Try to get goal position from goal_map first (shows current navigation target)
    goal_row, goal_col = None, None

    if hasattr(agent, 'goal_map') and agent.goal_map is not None:
        # Find position where goal_map == 1
        goal_positions = np.where(agent.goal_map == 1)
        if len(goal_positions[0]) > 0:
            # Use the first goal position found
            # Note: goal_map is already flipped (goal_map[::-1]), so we need to flip it back
            goal_row = goal_positions[0][0]
            goal_col = goal_positions[1][0]

            # Draw goal at this position
            map[:, goal_row-goal_size:goal_row+goal_size, goal_col-goal_size:goal_col+goal_size] = 0
            map[color_index, goal_row-goal_size:goal_row+goal_size, goal_col-goal_size:goal_col+goal_size] = 1
            return

    # Fallback 1: Use goal_loc if available (landmark-based navigation)
    if not agent.found_goal and agent.goal_loc is not None:
        # Convert goal_loc elements to int to ensure valid slicing
        goal_loc_0 = int(agent.goal_loc[0])
        goal_loc_1 = int(agent.goal_loc[1])
        map[:,int(agent.map_size_cm/5)-goal_loc_0-goal_size:int(agent.map_size_cm/5)-goal_loc_0+goal_size, goal_loc_1-goal_size:goal_loc_1+goal_size] = 0
        map[color_index,int(agent.map_size_cm/5)-goal_loc_0-goal_size:int(agent.map_size_cm/5)-goal_loc_0+goal_size, goal_loc_1-goal_size:goal_loc_1+goal_size] = 1
    # Fallback 2: Use goal_gps (final goal position)
    else:
        map[:, int((agent.map_size_cm/200+agent.goal_gps[1])*100/agent.resolution)-goal_size:int((agent.map_size_cm/200+agent.goal_gps[1])*100/agent.resolution)+goal_size, int((agent.map_size_cm/200+agent.goal_gps[0])*100/agent.resolution)-goal_size:int((agent.map_size_cm/200+agent.goal_gps[0])*100/agent.resolution)+goal_size] = 0
        map[color_index, int((agent.map_size_cm/200+agent.goal_gps[1])*100/agent.resolution)-goal_size:int((agent.map_size_cm/200+agent.goal_gps[1])*100/agent.resolution)+goal_size, int((agent.map_size_cm/200+agent.goal_gps[0])*100/agent.resolution)-goal_size:int((agent.map_size_cm/200+agent.goal_gps[0])*100/agent.resolution)+goal_size] = 1



def draw_landmark(agent, map, landmark_size, color_index=None, frontier_locations=None, frontier_color=(1.0, 1.0, 0.0)):
    # create a disk structuring element (not used directly but kept for parity with draw_goal)
    skimage.morphology.disk(landmark_size)

    # map is assumed to have shape (C, H, W)
    H = map.shape[1]
    W = map.shape[2]

    # Draw frontiers first (so landmarks are drawn on top)
    if frontier_locations is not None and len(frontier_locations) > 0:
        # Convert to numpy if it's a torch tensor
        if torch.is_tensor(frontier_locations):
            fl_np = frontier_locations.cpu().numpy()
        else:
            fl_np = frontier_locations

        frontier_size = 1
        for frontier in fl_np:
            try:
                r = int(frontier[0])
                c = int(frontier[1])
            except Exception:
                continue

            # Use same coordinate transform as landmarks
            top = int(agent.map_size_cm/5) - r - frontier_size
            bottom = int(agent.map_size_cm/5) - r + frontier_size
            left = c - frontier_size
            right = c + frontier_size

            # clip to map bounds
            top = max(0, top)
            left = max(0, left)
            bottom = min(H, bottom)
            right = min(W, right)

            if top >= bottom or left >= right:
                continue

            # Set RGB color for frontiers
            map[0, top:bottom, left:right] = frontier_color[0]
            map[1, top:bottom, left:right] = frontier_color[1]
            map[2, top:bottom, left:right] = frontier_color[2]

    # If there are no landmarks, return after drawing frontiers
    if not agent.landmarks:
        return

    # Helper function to transform landmark coordinates to map coordinates
    def transform_coords(r, c):
        # Ensure all values are converted to native Python int
        y = int(int(agent.map_size_cm/5) - int(r))
        x = int(c)
        return (y, x)

    # Draw edges between landmarks (if available from landmark_map)
    if hasattr(agent, 'landmark_map') and agent.landmark_map is not None:
        edges = agent.landmark_map.get_edges()
        if edges is not None and len(edges) > 0:
            import cv2
            # Convert map to numpy for drawing lines
            # Use .copy() to ensure contiguous memory layout for OpenCV
            map_np = map.permute(1, 2, 0).cpu().numpy().copy()  # (H, W, C)
            # Ensure the array is contiguous and in the correct format
            map_np = np.ascontiguousarray(map_np, dtype=np.float32)

            for edge in edges:
                (r1, c1), (r2, c2) = edge
                y1, x1 = transform_coords(r1, c1)
                y2, x2 = transform_coords(r2, c2)

                # Convert to native Python int for OpenCV
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)

                # Clip to bounds
                if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                    # Draw cyan edges for topological graph
                    cv2.line(map_np, (x1, y1), (x2, y2), (0.0, 1.0, 1.0), thickness=3)

            # Convert back to tensor
            map[:, :, :] = torch.from_numpy(map_np).permute(2, 0, 1)

    # decide which color channel to use for landmarks
    if color_index is None:
        color_idx = COLOR_LANDMARK
    else:
        try:
            color_idx = int(color_index)
        except Exception:
            color_idx = COLOR_LANDMARK

    # Draw landmarks with different styles
    for idx, lm in enumerate(agent.landmarks):
        # Expect each landmark as (row, col) in the same local-map coordinates as agent.goal_loc
        try:
            r = int(lm[0])
            c = int(lm[1])
        except Exception:
            # skip malformed entries
            continue

        top = int(agent.map_size_cm/5) - r - landmark_size
        bottom = int(agent.map_size_cm/5) - r + landmark_size
        left = c - landmark_size
        right = c + landmark_size

        # clip to map bounds
        top = max(0, top)
        left = max(0, left)
        bottom = min(H, bottom)
        right = min(W, right)

        if top >= bottom or left >= right:
            continue

        # Color based on type (if semantic info available)
        if hasattr(agent, 'landmark_map') and agent.landmark_map is not None:
            semantic_info = agent.landmark_map.semantic_info
            if semantic_info and idx < len(semantic_info):
                lm_type = semantic_info[idx].get('type', 'other')
                if lm_type == 'intersection':
                    # Intersection: blue
                    map[:, top:bottom, left:right] = 0
                    map[0, top:bottom, left:right] = 0.0  # No red
                    map[1, top:bottom, left:right] = 0.0  # No green
                    map[2, top:bottom, left:right] = 1.0  # Full blue
                elif lm_type == 'leaf':
                    # Leaf: blue (same as intersection)
                    map[:, top:bottom, left:right] = 0
                    map[0, top:bottom, left:right] = 0.0  # No red
                    map[1, top:bottom, left:right] = 0.0  # No green
                    map[2, top:bottom, left:right] = 1.0  # Full blue
                else:
                    # Other: blue (default)
                    map[:, top:bottom, left:right] = 0
                    map[color_idx, top:bottom, left:right] = 1
            else:
                # Default: blue
                map[:, top:bottom, left:right] = 0
                map[color_idx, top:bottom, left:right] = 1
        else:
            # Default: blue
            map[:, top:bottom, left:right] = 0
            map[color_idx, top:bottom, left:right] = 1


def draw_frontiers(agent, map, frontier_locations, frontier_size=1, color=(1.0, 1.0, 0.0)):
    """Draw frontier points on the map with a custom color (default: yellow).

    Args:
        agent: The agent instance
        map: Map tensor with shape (C, H, W)
        frontier_locations: Tensor or array of shape (N, 2) with (row, col) coordinates
        frontier_size: Size of each frontier point marker
        color: RGB color tuple (values in 0-1 range), default yellow (1, 1, 0)
    """
    if frontier_locations is None or len(frontier_locations) == 0:
        return

    # map is assumed to have shape (C, H, W)
    H = map.shape[1]
    W = map.shape[2]

    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(frontier_locations):
        frontier_locations = frontier_locations.cpu().numpy()

    for frontier in frontier_locations:
        try:
            # frontier_locations are already in map coordinates (row, col)
            r = int(frontier[0])
            c = int(frontier[1])
        except Exception:
            continue

        top = r - frontier_size
        bottom = r + frontier_size + 1
        left = c - frontier_size
        right = c + frontier_size + 1

        # clip to map bounds
        top = max(0, top)
        left = max(0, left)
        bottom = min(H, bottom)
        right = min(W, right)

        if top >= bottom or left >= right:
            continue

        # Set RGB color directly
        map[0, top:bottom, left:right] = color[0]  # R channel
        map[1, top:bottom, left:right] = color[1]  # G channel
        map[2, top:bottom, left:right] = color[2]  # B channel


def draw_bounding_boxes_with_captions(image, xyxy_list, captions, confidences=None):
    """Draw bounding boxes with captions on an RGB image.

    Args:
        image: RGB image array (H, W, 3) in range [0, 255]
        xyxy_list: List or array of bounding boxes in xyxy format (N, 4)
        captions: List of caption strings for each box
        confidences: Optional list of confidence scores

    Returns:
        Annotated image with bounding boxes and captions
    """
    # Make a copy to avoid modifying the original
    annotated_image = image.copy()

    if len(xyxy_list) == 0 or len(captions) == 0:
        return annotated_image

    # Generate distinct colors for each caption
    unique_captions = sorted(set(captions))
    color_map = {}

    # Use diverse colors
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Spring Green
        (255, 0, 128),    # Rose
    ]

    for i, caption in enumerate(unique_captions):
        color_map[caption] = colors[i % len(colors)]

    # Draw each bounding box
    for i, (bbox, caption) in enumerate(zip(xyxy_list, captions)):
        # Get bbox coordinates
        x1, y1, x2, y2 = map(int, bbox)

        # Get color for this caption
        color = color_map.get(caption, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        if confidences is not None and i < len(confidences):
            label = f"{caption} ({confidences[i]:.2f})"
        else:
            label = caption

        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw background rectangle for text
        cv2.rectangle(annotated_image,
                     (x1, y1 - text_height - baseline - 5),
                     (x1 + text_width, y1),
                     color,
                     -1)  # Filled rectangle

        # Draw text
        cv2.putText(annotated_image,
                   label,
                   (x1, y1 - baseline - 2),
                   font,
                   font_scale,
                   (255, 255, 255),  # White text
                   thickness,
                   cv2.LINE_AA)

    return annotated_image


def generate_caption_colors(captions):
    """Generate a color mapping for unique captions.

    Args:
        captions: List of caption strings

    Returns:
        Dictionary mapping caption -> RGB color tuple (values in 0-1 range)
    """
    import matplotlib.pyplot as plt

    unique_captions = sorted(set(captions))

    if len(unique_captions) == 0:
        return {}

    # Use a colormap with many distinct colors
    if len(unique_captions) <= 10:
        cmap = plt.cm.get_cmap('tab10')
    elif len(unique_captions) <= 20:
        cmap = plt.cm.get_cmap('tab20')
    else:
        # For many captions, use HSV for maximum distinction
        cmap = plt.cm.get_cmap('hsv')

    color_map = {}
    for i, caption in enumerate(unique_captions):
        if len(unique_captions) <= 20:
            color = cmap(i)
        else:
            # Spread colors evenly across the HSV spectrum
            color = cmap(i / len(unique_captions))
        color_map[caption] = color[:3]  # RGB only, ignore alpha

    return color_map


def draw_objects_by_caption(agent, map_tensor, dilation_size=1):
    """Draw scene graph objects on the map, colored by their captions.

    Args:
        agent: The agent instance containing scenegraph
        map_tensor: Map tensor with shape (C, H, W) in PyTorch format
        dilation_size: Size of morphological dilation to apply (0 for no dilation)
    """
    if not hasattr(agent, 'scenegraph') or not hasattr(agent.scenegraph, 'nodes'):
        return

    nodes = agent.scenegraph.nodes
    if len(nodes) == 0:
        return

    # Collect all captions
    captions = [node.caption for node in nodes if node.caption is not None]
    if len(captions) == 0:
        return

    # Generate color mapping
    color_map = generate_caption_colors(captions)

    # Draw each node with its caption color using point cloud data
    H, W = map_tensor.shape[1], map_tensor.shape[2]
    map_resolution = agent.map_resolution
    map_size = agent.map_size

    for node in nodes:
        if node.caption is None:
            continue

        try:
            # Get the object's point cloud
            if not hasattr(node, 'object') or node.object is None:
                continue

            if 'pcd' not in node.object or node.object['pcd'] is None:
                continue

            # Extract all points from the point cloud
            points = np.asarray(node.object['pcd'].points)

            if len(points) == 0:
                continue

            # Convert 3D points to 2D map coordinates
            # Following the same logic as in core_func.py update_node()
            points_x = (points[:, 0] * 100 / map_resolution).astype(int)
            points_y = (points[:, 1] * 100 / map_resolution).astype(int)
            points_y = map_size - 1 - points_y  # Flip y-axis

            # Filter points within map bounds
            valid_mask = (points_x >= 0) & (points_x < W) & (points_y >= 0) & (points_y < H)
            points_x = points_x[valid_mask]
            points_y = points_y[valid_mask]

            if len(points_x) == 0:
                continue

            # Get color for this caption
            color_rgb = color_map.get(node.caption, (0.5, 0.5, 0.5))

            # Create a binary mask for this object
            object_mask = np.zeros((H, W), dtype=bool)
            object_mask[points_y, points_x] = True

            # Apply morphological dilation to make objects more visible
            if dilation_size > 0:
                object_mask = skimage.morphology.binary_dilation(
                    object_mask,
                    skimage.morphology.disk(dilation_size)
                )

            # Set the color for all points in the object
            map_tensor[0, object_mask] = color_rgb[0]
            map_tensor[1, object_mask] = color_rgb[1]
            map_tensor[2, object_mask] = color_rgb[2]

        except Exception:
            # Skip nodes with invalid data
            continue


def draw_landmarks_on_rgb_image(image, agent, edge_color=(0, 255, 255), edge_thickness=1):
    """Draw landmarks and edges on an RGB image (numpy array).

    This function draws the landmark map visualization on any RGB image.
    Landmarks are colored by type (intersection/leaf), and edges are drawn between them.

    Args:
        image: RGB image as numpy array (H, W, 3) in uint8 format
        agent: Agent instance containing landmark_map
        edge_color: Color for edges in RGB format (default: cyan (0, 255, 255))
        edge_thickness: Thickness of edge lines (default: 1)

    Returns:
        Modified image with landmarks drawn on it
    """
    if not hasattr(agent, 'landmark_map') or agent.landmark_map is None:
        return image

    if not hasattr(agent, 'landmarks') or not agent.landmarks:
        return image

    # Ensure the image is contiguous for OpenCV operations
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    H, W = image.shape[:2]

    # Helper function to transform landmark coordinates to map coordinates
    # Matches the transformation used in draw_landmark function
    def transform_coords(r, c):
        # Ensure all values are converted to native Python int
        y = int(int(agent.map_size_cm/5) - int(r))
        x = int(c)
        return (x, y)  # Return (x, y) for cv2 drawing functions

    # Draw edges first (so they appear below nodes)
    edges = agent.landmark_map.get_edges()
    if edges is not None and len(edges) > 0:
        for edge in edges:
            (r1, c1), (r2, c2) = edge
            x1, y1 = transform_coords(r1, c1)
            x2, y2 = transform_coords(r2, c2)

            # Convert to native Python int for OpenCV
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

            # Clip to bounds
            if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                cv2.line(image, (x1, y1), (x2, y2), edge_color, thickness=edge_thickness)

    # Get semantic info for coloring landmarks by type
    semantic_info = agent.landmark_map.semantic_info if hasattr(agent.landmark_map, 'semantic_info') else None

    # Draw landmarks
    for idx, lm in enumerate(agent.landmarks):
        try:
            r = int(lm[0])
            c = int(lm[1])
        except Exception:
            continue

        x, y = transform_coords(r, c)

        # Convert to native Python int for OpenCV
        x, y = int(x), int(y)

        # Check bounds
        if not (0 <= x < W and 0 <= y < H):
            continue

        # Determine landmark color based on type
        # Note: Colors are in RGB format to match the input image format
        if semantic_info and idx < len(semantic_info):
            lm_type = semantic_info[idx].get('type', 'other')
            if lm_type == 'intersection':
                # Intersection: blue
                color = (0, 0, 255)  # RGB: blue
            elif lm_type == 'leaf':
                # Leaf: blue (same as intersection)
                color = (0, 0, 255)  # RGB: blue
            else:
                # Other: blue
                color = (0, 0, 255)  # RGB: blue
        else:
            # Default: blue
            color = (0, 0, 255)  # RGB: blue

        radius = 8  # Increased for better visibility in full occupancy map

        # Draw landmark circle with white border for visibility
        cv2.circle(image, (x, y), radius, color, -1)
        cv2.circle(image, (x, y), radius, (255, 255, 255), 2)

    return image
