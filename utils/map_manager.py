"""
Map Manager - Handles all map-related operations for PSG-Nav agent.

This module extracts map initialization, updating, and saving functionality
from the main PSG_Nav_Agent class to improve code organization.
"""

import os
import copy
import numpy as np
import torch
import cv2
import skimage
import utils.utils_fmm.pose_utils as pu


class MapManager:
    """Manages all map-related operations including room maps, free maps, and traversibility."""

    def __init__(self, agent):
        """
        Initialize MapManager.

        Args:
            agent: Reference to the main PSG_Nav_Agent instance
        """
        self.agent = agent
        self.device = agent.device
        self.map_resolution = agent.map_resolution
        self.map_size_cm = agent.map_size_cm

        # Scene name mapping: 友好名称 -> 实际scene_id编码
        # 方便配置时使用易读的名称，如 'scene_002' 而不是 '6s7QHgap2fW'
        self.scene_name_mapping = {
            'scene_002': '6s7QHgap2fW',
            'scene_012': 'p53SfW6mjZe',
            'scene_013': 'p53SfW6mjZe',
            
            # 继续添加其他场景的映射
            # 'scene_001': 'actual_scene_id_001',
            # 'scene_003': 'actual_scene_id_003',
        }

        # Scene-specific disk radius configuration for traversibility inflation
        # Map format: 'scene_id': disk_radius or 'scene_id/episode_id': disk_radius
        # Default is 2, add exceptions for scenes that work better with 1
        # 可以使用友好名称（如'scene_002'）或实际编码（如'6s7QHgap2fW'），支持两种方式
        self.disk_radius_config = {
            # Add scene IDs or scene/episode combinations that need disk(1)
            # Examples:
            # 'RoyxdfgMDEo': 1,              # 整个场景都用disk(1)
            # '6s7QHgap2fW/0': 1,            # 使用实际编码
            # 'scene_002/0': 1,              # 使用友好名称（更易读）
            'scene_002/0': 1,
            'scene_002/2': 1,
            'scene_002/45': 1,
            
        }
        self.default_disk_radius = 2
        self.prev_agent_map_pos = None

    def init_map(self):
        """Initialize all maps (full_map, room_map, visited, collision, etc.)."""
        self.agent.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = self.agent.map_size, self.agent.map_size

        self.agent.full_map = torch.zeros(1, 1, full_w, full_h).float().to(self.device)
        self.agent.room_map = torch.zeros(1, 9, full_w, full_h).float().to(self.device)
        self.agent.visited = self.agent.full_map[0, 0].cpu().numpy()
        self.agent.collision_map = self.agent.full_map[0, 0].cpu().numpy()
        self.agent.fbe_free_map = copy.deepcopy(self.agent.full_map).to(self.device)
        self.agent.full_pose = torch.zeros(3).float().to(self.device)
        self.agent.goal_gps_map = self.agent.full_map[0, 0].cpu().numpy()
        self.agent.goal_gps_timestamp_map = np.zeros((full_w, full_h), dtype=np.int32)  # Track last detection step
        self.agent.origins = np.zeros((2))

        # Agent trajectory tracking map
        self.agent.agent_trajectory_map = np.zeros((full_w, full_h), dtype=np.uint8)

        # Initialize disk radius for current episode (called once per episode for performance)
        self.agent.disk_radius = 2
        def init_map_and_pose():
            self.agent.full_map.fill_(0.)
            self.agent.full_pose.fill_(0.)
            self.agent.full_pose[:2] = self.map_size_cm / 100.0 / 2.0
            self.agent.agent_trajectory_map.fill(0)

        init_map_and_pose()

    def update_map(self, observations):
        """Update the main obstacle map."""
        self.agent.full_pose[0] = self.map_size_cm / 100.0 / 2.0 + torch.from_numpy(observations['gps']).to(self.device)[0]
        self.agent.full_pose[1] = self.map_size_cm / 100.0 / 2.0 - torch.from_numpy(observations['gps']).to(self.device)[1]
        self.agent.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device)
        self.agent.full_map = self.agent.sem_map_module(
            torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device),
            self.agent.full_pose,
            self.agent.full_map
        )

    def update_free_map(self, observations):
        """Update the free space map."""
        self.agent.full_pose[0] = self.map_size_cm / 100.0 / 2.0 + torch.from_numpy(observations['gps']).to(self.device)[0]
        self.agent.full_pose[1] = self.map_size_cm / 100.0 / 2.0 - torch.from_numpy(observations['gps']).to(self.device)[1]
        self.agent.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device)
        self.agent.fbe_free_map = self.agent.free_map_module(
            torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device),
            self.agent.full_pose,
            self.agent.fbe_free_map
        )

        agent_x = int(self.agent.full_pose[0].item() * 100 / self.map_resolution)
        agent_y = int((self.map_size_cm / 100 - self.agent.full_pose[1].item()) * 100 / self.map_resolution)
        theta = np.deg2rad(self.agent.full_pose[2].item())

        free_map_np = self.agent.fbe_free_map.detach().cpu().numpy()
        map_h, map_w = free_map_np.shape[-2], free_map_np.shape[-1]

        def carve_square(row, col, half_size):
            r0 = max(0, row - half_size)
            r1 = min(map_h, row + half_size + 1)
            c0 = max(0, col - half_size)
            c1 = min(map_w, col + half_size + 1)
            free_map_np[r0:r1, c0:c1] = 1

        # Mark the robot footprint itself as free.
        carve_square(agent_y, agent_x, 3)

        # Connect consecutive robot positions so traversed space stays free.
        if self.prev_agent_map_pos is not None:
            prev_row, prev_col = self.prev_agent_map_pos
            cv2.line(free_map_np, (prev_col, prev_row), (agent_x, agent_y), color=1, thickness=3)

        # Fill the short blind region under and just in front of the depth camera.
        forward_dx = np.cos(theta)
        forward_dy = -np.sin(theta)
        patch_half_width = 2
        for step_px in range(1, 7):
            row = int(round(agent_y + forward_dy * step_px))
            col = int(round(agent_x + forward_dx * step_px))
            carve_square(row, col, patch_half_width)

        self.prev_agent_map_pos = (agent_y, agent_x)
        self.agent.fbe_free_map = torch.from_numpy(free_map_np).to(self.device)

    def update_room_map(self, observations, room_prediction_result):
        """
        Update room map based on GLIP detection results.

        Args:
            observations: Current observations including depth
            room_prediction_result: GLIP detection results for rooms
        """
        new_room_labels = self.agent.detection_manager._get_glip_real_label(room_prediction_result)

        # Import rooms list
        from utils.utils_glip import rooms

        type_mask = np.zeros((9, self.agent.config.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                             self.agent.config.SIMULATOR.DEPTH_SENSOR.WIDTH))
        bboxs = room_prediction_result.bbox
        score_vec = torch.zeros((9)).to(self.device)

        for i, box in enumerate(bboxs):
            box = box.to(torch.int64)
            idx = rooms.index(new_room_labels[i])
            type_mask[idx, box[1]:box[3], box[0]:box[2]] = 1
            score_vec[idx] = room_prediction_result.get_field("scores")[i]

        # Update room_map
        self.agent.room_map = self.agent.room_map_module(
            torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device),
            self.agent.full_pose,
            self.agent.room_map,
            torch.from_numpy(type_mask).to(self.device).type(torch.float32),
            score_vec
        )

    def get_traversible(self, map_pred, pose_pred):
        """
        Get traversible area from map and update visited regions.

        Args:
            map_pred: Map prediction tensor
            pose_pred: Pose prediction (start_x, start_y, start_o, gx1, gx2, gy1, gy2)

        Returns:
            tuple: (traversible, start, start_o)
        """
        grid = np.rint(map_pred)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [int(r * 100 / self.map_resolution - gy1),
                 int(c * 100 / self.map_resolution - gx1)]
        start = pu.threshold_poses(start, grid.shape)

        self.agent.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                              start[1]-2:start[1]+3] = 1
        abs_r = start[0] + gy1
        abs_c = start[1] + gx1
        self.agent.agent_trajectory_map[max(0, abs_r-2):min(self.agent.map_size, abs_r+3),
                                        max(0, abs_c-2):min(self.agent.map_size, abs_c+3)] = 1

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2, w+2)) + value
            new_mat[1:h+1, 1:w+1] = mat
            return new_mat

        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1 = 0, 0
        x2, y2 = grid.shape

        traversible = skimage.morphology.binary_dilation(
            grid[y1:y2, x1:x2],
            self.agent.selem) != True

        if not (traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.agent.navigate_steps)

        traversible = 1 - traversible
        # Increase inflation radius to prevent getting stuck at corners
        # Use scene-specific disk radius (initialized once per episode in init_map)
        selem = skimage.morphology.disk(self.agent.disk_radius)
        traversible = skimage.morphology.binary_dilation(traversible, selem)
        traversible[self.agent.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = skimage.morphology.binary_dilation(traversible, selem) != True

        traversible[int(start[0]-y1)-1:int(start[0]-y1)+2,
                    int(start[1]-x1)-1:int(start[1]-x1)+2] = 1
        traversible = traversible * 1.

        traversible[self.agent.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = add_boundary(traversible)

        return traversible, start, start_o

    def save_room_map_with_grid(self, room_map_np, room_names):
        """
        Save room map visualization with coordinate grid.

        Args:
            room_map_np: Room map numpy array [9, H, W]
            room_names: List of room names
        """
        # Room colors (BGR format, matching visualizer.py)
        room_colors_bgr = {
            0: (193, 182, 255),   # bedroom - light pink
            1: (144, 238, 144),   # living room - light green
            2: (230, 216, 173),   # bathroom - light blue
            3: (185, 218, 255),   # kitchen - peach
            4: (221, 160, 221),   # dining room - plum
            5: (140, 230, 240),   # office room - khaki
            6: (122, 160, 255),   # gym - light salmon
            7: (230, 224, 176),   # lounge - powder blue
            8: (211, 211, 211),   # laundry room - darker gray
        }

        # room_map_np shape: [9, H, W]
        H, W = room_map_np.shape[1], room_map_np.shape[2]

        # Create RGB visualization
        room_vis = np.full((H, W, 3), 255, dtype=np.uint8)  # White background

        # Get the dominant room type at each position
        room_sum = room_map_np.sum(axis=0)
        has_room_info = room_sum > 0.1

        if has_room_info.any():
            dominant_room = room_map_np.argmax(axis=0)

            # Color each position by its dominant room type
            for room_idx, color in room_colors_bgr.items():
                mask = (dominant_room == room_idx) & has_room_info
                room_vis[::-1][mask] = color

        # Draw coordinate grid
        grid_spacing = 50
        grid_color = (180, 180, 180)
        grid_thickness = 1

        # Vertical grid lines
        for x in range(0, W, grid_spacing):
            cv2.line(room_vis, (x, 0), (x, H), grid_color, grid_thickness)
            if x % (grid_spacing * 2) == 0:
                cv2.putText(room_vis, str(x), (x + 2, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # Horizontal grid lines
        for y in range(0, H, grid_spacing):
            cv2.line(room_vis, (0, y), (W, y), grid_color, grid_thickness)
            if y % (grid_spacing * 2) == 0:
                cv2.putText(room_vis, str(y), (5, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw coordinate axes
        map_center_x = W // 2
        map_center_y = H // 2
        axis_color = (100, 100, 100)
        axis_thickness = 2

        cv2.line(room_vis, (0, map_center_y), (W, map_center_y), axis_color, axis_thickness)
        cv2.line(room_vis, (map_center_x, 0), (map_center_x, H), axis_color, axis_thickness)

        cv2.putText(room_vis, f"X=0", (map_center_x + 5, map_center_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(room_vis, f"Y=0", (map_center_x + 5, map_center_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Save the image
        save_dir = os.path.join(self.agent.visualization_dir, 'room_maps')
        os.makedirs(save_dir, exist_ok=True)

        timestep = getattr(self.agent, 'timestep', getattr(self.agent, 'total_steps', 0))
        filename = f'room_map_step_{timestep:04d}_ep_{self.agent.count_episodes:02d}.png'
        save_path = os.path.join(save_dir, filename)

        cv2.imwrite(save_path, room_vis)
        print(f"已保存 Room Map 图像: {save_path}")
