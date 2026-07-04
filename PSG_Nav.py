import warnings
# 禁用特定的警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*requires_grad.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*upsample_bilinear.*deprecated.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*device.*argument.*deprecated.*')

import argparse
import copy
import math
import os
from pathlib import Path
import numpy as np
import skimage
import torch
import habitat

from GLIP.maskrcnn_benchmark.config import cfg as glip_cfg
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

# Import unified object detector
from configs.detector_config import DETECTOR_TYPE
from configs.dataset_registry import (
    SUPPORTED_DATASETS,
    get_dataset_config_path,
    get_default_dataset_split,
)
from utils.object_detector import create_detector

from graph.graph_class import SceneGraph
from graph.enumerated_groups import EnumeratedGroupNodes
from graph.room_group_combinations import RoomGroupCombinations
from graph.scenegraph_combinations import SceneGraphCombinations

from utils.utils_frontiers import calculate_frontiers
from utils.landmark_map import LandmarkMap
from utils.llm_filter import LLMFilter

import utils.utils_fmm.control_helper as CH
import utils.utils_fmm.pose_utils as pu
from utils.utils_fmm.fmm_planner import FMMPlanner
from utils.utils_fmm.mapping import Semantic_Mapping
from utils.utils_glip import *
from visualizations import visualize as _visualize
from utils.collect_scene_mappings import log_scene_info

# Import new modular managers
from utils.map_manager import MapManager
from utils.landmark_manager import LandmarkManager
from utils.landmark_selector import LandmarkSelector
from utils.navigation_planner import NavigationPlanner
from utils.object_detection import ObjectDetectionManager



class PSG_Nav_Agent():
    def __init__(self, task_config, args=None):
        self.config = task_config
        self.args = args

        # Initialize dataset-specific categories FIRST (before any detector/scenegraph init)
        # This must happen before GLIP/FastSAM initialization
        dataset_name = args.dataset if args and hasattr(args, 'dataset') else 'hssd'
        print(f"\n{'='*60}")
        print(f"[Dataset Init] Initializing PSG-Nav for dataset: {dataset_name.upper()}")
        print(f"{'='*60}\n")

        # Import category initialization
        from utils.utils_glip import init_categories
        from configs.dataset_categories import dataset_to_unified_name

        # Initialize categories for current dataset
        init_categories(dataset_name)

        # Store dataset for later use
        self.dataset = dataset_name
        self.dataset_to_unified_name = dataset_to_unified_name

        self.panoramic = []
        self.panoramic_depth = []
        self.device = (
            torch.device("cuda:{}".format(0)) # change gpu
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.prev_action = 0
        self.navigate_steps = 0
        self.move_steps = 0
        self.total_steps = 0
        self.landmark_navigation_steps = 0  # Track steps since selecting current landmark
        self.max_landmark_navigation_steps = 50  # Re-call FBE if landmark not reached in 50 steps
        self.found_goal = False
        self.found_goal_times = 0
        self.distance_threshold = 4
        self.former_collide = 0
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = -1
        self.loop_time = 0
        self.last_segment_num = 0
        self.goal_merge_threshold = 0.8
        self.rooms = rooms
        self.rooms_captions = rooms_captions

        self.split = (self.args.split_l >= 0)
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
        
        # landmark
        self.landmarks = []

        ### ------ init object detector (GLIP or FastSAM) ------ ###
        print(f"[Detector] Initializing {DETECTOR_TYPE.upper()} detector...")

        if DETECTOR_TYPE == 'glip':
            # Use GLIP detector (original method)
            config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml"
            weight_file = "GLIP/MODEL/glip_large_model.pth"
            glip_confidence_threshold = (
                float(args.glip_confidence_threshold)
                if args is not None and hasattr(args, "glip_confidence_threshold")
                else 0.61
            )
            glip_cfg.local_rank = 0
            glip_cfg.num_gpus = 1
            glip_cfg.merge_from_file(config_file)
            glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
            glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
            self.glip_demo = GLIPDemo(
                glip_cfg,
                min_image_size=800,
                confidence_threshold=glip_confidence_threshold,
                show_mask_heatmaps=False
            )
            self.detector = None  # Use glip_demo directly for backward compatibility

        elif DETECTOR_TYPE == 'fastsam_clip':
            # Use FastSAM + CLIP detector
            from configs.detector_config import FASTSAM_CLIP_CONFIG
            self.detector = create_detector('fastsam_clip', **FASTSAM_CLIP_CONFIG)
            self.glip_demo = None  # Not using GLIP

        elif DETECTOR_TYPE == 'fastsam_text':
            # Use FastSAM + text_prompt detector
            from configs.detector_config import FASTSAM_TEXT_CONFIG
            self.detector = create_detector('fastsam_text', **FASTSAM_TEXT_CONFIG)
            self.glip_demo = None  # Not using GLIP

        else:
            raise ValueError(f"Unknown detector type: {DETECTOR_TYPE}. Choose 'glip', 'fastsam_clip', or 'fastsam_text'")

        print(f"[Detector] {DETECTOR_TYPE.upper()} detector initialized successfully!")



        self.map_size_cm = 4000
        self.resolution = self.map_resolution = 5
        self.map_height = int(self.map_size_cm / self.map_resolution)
        self.camera_horizon = 0
        self.dilation_deg = 0
        self.collision_threshold = 0.08
        self.selem = skimage.morphology.square(1)
        self.explanation = ''
        self.landmark_idx = None
        # Stuck recovery state (for forced action sequence recovery)
        self.in_stuck_recovery = False
        self.orientation_history = []  # Record recent orientations (for stuck recovery)
        self.orientation_history_size = 10  # Keep last 10 frames
        self.stuck_recovery_target_angle = None  # Target orientation for stuck recovery
        self.stuck_recovery_phase = None  # "turning" or "forward" or None

        # Initialize modular managers
        self.map_manager = MapManager(self)
        self.landmark_manager = LandmarkManager(self)
        self.landmark_selector = LandmarkSelector(self)
        self.navigation_planner = NavigationPlanner(self)
        self.detection_manager = ObjectDetectionManager(self)

        self.init_map()
        self.sem_map_module = Semantic_Mapping(self).to(self.device) 
        self.free_map_module = Semantic_Mapping(self, max_height=10,min_height=-150).to(self.device)
        self.room_map_module = Semantic_Mapping(self, max_height=200,min_height=-10, num_cats=9).to(self.device)
        
        self.free_map_module.eval()
        self.free_map_module.set_view_angles(self.camera_horizon)
        self.sem_map_module.eval()
        self.sem_map_module.set_view_angles(self.camera_horizon)
        self.room_map_module.eval()
        self.room_map_module.set_view_angles(self.camera_horizon)

        self.camera_matrix = self.free_map_module.camera_matrix
        self.co_occur_mtx = np.load('tools/obj.npy')
        self.co_occur_mtx -= self.co_occur_mtx.min()
        self.co_occur_mtx /= self.co_occur_mtx.max() 
        
        self.co_occur_room_mtx = np.load('tools/room.npy')
        self.co_occur_room_mtx -= self.co_occur_room_mtx.min()
        self.co_occur_room_mtx /= self.co_occur_room_mtx.max()

        self.scenegraph = SceneGraph(map_resolution=self.map_resolution, map_size_cm=self.map_size_cm, map_size=self.map_size, camera_matrix=self.camera_matrix, agent=self, server_port=self.args.server_port, dataset=self.dataset)

        self.experiment_name = 'experiment_0'

        if self.split:
            self.experiment_name = self.experiment_name + f'/[{self.args.split_l}:{self.args.split_r}]'

        # Add timestamp suffix to distinguish different runs (format: MMDD_HHMM)
        # Use provided timestamp (for parallel runs) or generate new one
        from datetime import datetime
        if hasattr(self.args, 'timestamp') and self.args.timestamp:
            self.timestamp = self.args.timestamp
        else:
            self.timestamp = datetime.now().strftime("%m%d_%H%M")


        dataset_suffix = f"_{self.args.dataset}" if hasattr(self.args, 'dataset') and self.args.dataset else ""
        self.visualization_dir = f'data/visualization{dataset_suffix}_{self.timestamp}/{self.experiment_name}/'

        # landmark map manager (sparse landmark graph for frontiers)
        self.landmark_map = LandmarkMap(max_nodes=60, min_dist=5.0, knn=3, method="voronoi", save_visualization=False)
        # persistent copy of latest landmarks (similar to goal_loc)
        self.landmark_nodes = None
        self.landmark_edges = []

        # RAG latest goal detection tracking (only stores the most recent find_goal frame)
        self.rag_latest_goal_detection = None
        # RAG verification: store the document used for comparison
        self.rag_comparison_doc = None  # Stores the RAGDocument used in latest verification
        # RAG sliding window: keep latest N detections with matching caption
        self.rag_sliding_window = []  # List of detection dicts
        self.rag_sliding_window_size = args.rag_window_size if args is not None else 5
        # RAG check enabled flag for current episode
        self.rag_check_enabled = False

        # Track which detection path is used for current goal (for visualization)
        # 'navigation': using GLIP/FastSAM (large objects)
        # 'scenegraph': using GroundingDINO (small objects)
        self.active_detection_path = 'navigation'

        # RAG mode: 0=disabled, 1=collect_only, 2=use_only, 3=full
        self.rag_mode = args.rag_mode if args is not None else 3
        self.rag_use_enabled = self.rag_mode in [2, 3]  # Use RAG for verification
        self.rag_collect_enabled = self.rag_mode in [1, 3]  # Collect to RAG

        # Initialize RAG Manager with configurable hyperparameters
        from utils.rag_manager import RAGManager
        if self.rag_mode > 0:  # Only initialize if RAG is not disabled
            self.rag_manager = RAGManager(
                storage_dir="data/rag_storage/rag_positives",
                active_dataset=args.dataset if args is not None else None,
                max_docs_per_category=args.rag_max_docs if args is not None else 20,
                similarity_threshold=args.rag_similarity_threshold if args is not None else 0.7,
                caption_penalty=args.rag_caption_penalty if args is not None else 0.3
            )
        else:
            self.rag_manager = None

        # Track unreachable landmarks to avoid revisiting them
        self.unreachable_landmarks = set()  # Set of (row, col) tuples
        self.unreachable_landmark_radius = 5  # Exclusion radius in pixels

        print('scene graph module init finish!!!')

    def get_detector_confidence_threshold(self):
        """
        Get the confidence threshold of the current detector.
        Returns the threshold value used by GLIP or FastSAM detector.
        """
        from configs.detector_config import DETECTOR_TYPE

        if DETECTOR_TYPE == 'glip':
            if self.args is not None and hasattr(self.args, "glip_confidence_threshold"):
                return float(self.args.glip_confidence_threshold)
            return 0.61  # GLIP threshold
        elif DETECTOR_TYPE == 'fastsam_clip':
            return 0.6  # FastSAM CLIP threshold
        elif DETECTOR_TYPE == 'fastsam_text':
            return 0.5  # FastSAM text threshold
        else:
            return 0.6  # Default threshold

    def get_goal_detection_threshold(self):
        """
        Get detection threshold for current goal from scenegraph.threshold_list.
        Returns minimum number of detections required to confirm goal.
        """
        if hasattr(self, 'scenegraph') and hasattr(self.scenegraph, 'threshold_list'):
            # Use obj_goal_sg for threshold lookup (same as used in scenegraph)
            threshold = self.scenegraph.threshold_list.get(self.obj_goal_sg, None)
            if threshold is not None:
                return threshold
            # Fallback: try original obj_goal
            threshold = self.scenegraph.threshold_list.get(self.obj_goal, None)
            if threshold is not None:
                return threshold
        # Default fallback: use cfg.obj_min_detections
        return self.scenegraph.cfg.obj_min_detections if hasattr(self, 'scenegraph') else 3

    def reset_goal_detection(self):
        """
        Reset goal detection state (found_goal, goal_gps_map, etc.)
        This should be called when:
        1. RAG verification rejects a detection
        2. Goal detection fails (no detections in current frame)
        3. Agent gets stuck while navigating to a goal

        This ensures that detecting a NEW goal instance requires 3 fresh detections,
        even if we previously detected a DIFFERENT instance of the same category.
        """
        self.found_goal = False
        self.found_goal_times = 0
        self.found_possible_goal = False
        self.goal_gps = np.array([0.,0.])
        self.possible_goal_temp_gps = np.array([0.,0.])
        if hasattr(self, 'goal_map'):
            self.goal_map.fill(0)
        # Critical: Clear positive goal counts to prevent different object instances.
        if hasattr(self, 'goal_gps_map'):
            self.goal_gps_map[self.goal_gps_map > 0] = 0
        if hasattr(self, 'goal_gps_timestamp_map'):
            self.goal_gps_timestamp_map.fill(0)
        # Clear RAG latest detection since we're resetting
        self.rag_latest_goal_detection = None

    def snap_goal_to_traversible(self, row: int, col: int, traversible: np.ndarray, max_radius: int = 15):
        """Snap a goal cell inside obstacle to the nearest traversible cell."""
        row = max(0, min(traversible.shape[0] - 1, int(row)))
        col = max(0, min(traversible.shape[1] - 1, int(col)))

        if traversible[row, col] > 0:
            return row, col

        best_pos = None
        best_dist2 = None
        for radius in range(1, max_radius + 1):
            y_start = max(0, row - radius)
            y_end = min(traversible.shape[0], row + radius + 1)
            x_start = max(0, col - radius)
            x_end = min(traversible.shape[1], col + radius + 1)

            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if traversible[y, x] <= 0:
                        continue
                    dist2 = (y - row) ** 2 + (x - col) ** 2
                    if best_dist2 is None or dist2 < best_dist2:
                        best_dist2 = dist2
                        best_pos = (y, x)

            if best_pos is not None:
                print(f"[Goal Snap] Goal map({row},{col}) is not traversible, snapped to map({best_pos[0]},{best_pos[1]})")
                return best_pos

        print(f"[Goal Snap] Goal map({row},{col}) is not traversible and no free cell found within radius {max_radius}")
        return row, col

    def reset(self):
        self.navigate_steps = 0
        self.move_steps = 0
        self.total_steps = 0
        self.landmark_navigation_steps = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.landmark_idx = None
        self.goal_loc = None
        self.prev_action = 0
        self.former_collide = 0
        self.goal_gps = np.array([0.,0.])
        self.possible_goal_temp_gps = np.array([0.,0.])
        self.last_gps = np.array([11100.,11100.])
        self.last_orientation = 0.0  # Track last orientation for stuck detection
        self.forward_no_move_count = 0  # Count consecutive forward actions without movement
        self.init_map()
        self.last_loc = self.full_pose
        self.panoramic = []
        self.panoramic_depth = []
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        self.found_possible_goal = False
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = self.count_episodes + 1
        self.loop_time = 0
        self.last_segment_num = 0
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}

        # Get goal from episode (dataset-specific naming)
        goal_from_episode = self.simulator._env.current_episode.object_category

        # Convert dataset-specific name to unified name (e.g., HSSD 'potted_plant' -> 'plant')
        self.obj_goal = self.dataset_to_unified_name(goal_from_episode, self.dataset)

        print(f"[Goal Conversion] Dataset goal: '{goal_from_episode}' -> Unified goal: '{self.obj_goal}'")

        # Set scene graph goal name (some objects use abbreviated names in SceneGraph)
        self.obj_goal_sg = self.obj_goal
        if self.obj_goal == 'gym_equipment':
            self.obj_goal_sg = 'treadmill. fitness equipment.'
        elif self.obj_goal == 'chest_of_drawers':
            self.obj_goal_sg = 'drawers'
        elif self.obj_goal == 'tv_monitor':
            self.obj_goal_sg = 'tv'

        # Get dataset and scene_id for RAG hierarchical storage
        self.current_dataset = self.args.dataset if hasattr(self.args, 'dataset') else "unknown"
        # Extract scene_id from episode.scene_id.
        if hasattr(self.simulator._env.current_episode, 'scene_id'):
            scene_id_full = self.simulator._env.current_episode.scene_id
            # Extract scene name from path (last part before .glb)
            scene_name = scene_id_full.split('/')[-1].replace('.glb', '')
            self.current_scene_id = scene_name
        else:
            self.current_scene_id = "unknown"

        print(f"[Episode {self.count_episodes}] Dataset: {self.current_dataset}, Scene: {self.current_scene_id}, Goal: {self.obj_goal}")

        # Initialize obj_locations based on dataset category count
        from configs.dataset_categories import get_category_count
        from utils.utils_glip import set_episode_goal_category

        set_episode_goal_category(self.obj_goal, self.dataset)

        num_categories = get_category_count(self.dataset, goal_category=self.obj_goal)
        self.obj_locations = [[] for i in range(num_categories)]
        print(f"[Obj Locations] Initialized {num_categories} categories for dataset {self.dataset.upper()}")

        self.current_obj_predictions = []

        self.rag_latest_goal_detection = None
        self.rag_comparison_doc = None
        self.rag_sliding_window = []  # Clear sliding window for new episode

        if self.rag_use_enabled and hasattr(self, 'rag_manager') and self.rag_manager is not None:
            self.rag_check_enabled = True
        else:
            self.rag_check_enabled = False
            if not self.rag_use_enabled:
                print(f"[RAG] Verification disabled by rag_mode={self.rag_mode}")

        self.not_move_steps = 0
        self.move_since_random = 0
        self.using_random_goal = False
        self.fronter_this_ex = 0
        self.random_this_ex = 0
        self.mode = 0
        # Track random navigation time periods
        self.random_nav_periods = []  # List of (start_step, end_step) tuples
        self.random_nav_start_step = None  # When current random nav started
        self.explanation = ''

        # Reset stuck recovery state
        self.in_stuck_recovery = False
        self.orientation_history = []
        self.stuck_recovery_target_angle = None
        self.stuck_recovery_phase = None

        self.scenegraph.reset()
        self.landmarks = []
        if hasattr(self, 'landmark_map'):
            self.landmark_map = LandmarkMap(max_nodes=60, min_dist=8.0, knn=3, method="voronoi", save_visualization=False)
        self.landmark_nodes = None
        self.landmark_edges = []

        log_scene_info(self)  

        # 清空临时的大对象
        if hasattr(self, 'landmark_sg_pairs'):
            self.landmark_sg_pairs = None
        if hasattr(self, 'enumerated_groups'):
            self.enumerated_groups = None
        if hasattr(self, 'room_combinations'):
            self.room_combinations = None

        # 清空 LLM 相关对象（它们内部可能有缓存）
        if hasattr(self, 'llm_filter'):
            del self.llm_filter

        # Reset unreachable landmarks tracking
        self.unreachable_landmarks.clear()

    def detect_objects(self, observations):
        """Delegate object detection to ObjectDetectionManager."""
        return self.detection_manager.detect_objects(observations)
                        
    def act(self, observations):
        if self.total_steps >= 500:
            return {"action": 0}
        number_action = 0
        self.total_steps += 1
        observations["depth"][observations["depth"]==0.5] = 100
        self.depth = observations["depth"]
        self.rgb = observations["rgb"][:,:,[2,1,0]]
        self.rgb_visualization = observations["rgb"]

        # Clear previous detection results at the start of each step
        # This prevents old masks from persisting in visualization when detection is not run
        self.current_obj_predictions = []

        self.scenegraph.set_agent(self)
        self.scenegraph.set_navigate_steps(self.navigate_steps)
        self.scenegraph.set_obj_goal(self.obj_goal, self.obj_goal_sg)
        self.scenegraph.set_room_map(self.room_map)
        self.scenegraph.set_fbe_free_map(self.fbe_free_map) # Frontier-based Exploration (occupancy map)
        self.scenegraph.set_observations(observations)
        self.scenegraph.set_full_map(self.full_map)
        self.scenegraph.set_full_pose(self.full_pose)
        self.scenegraph.update_scenegraph()
        
        self.update_map(observations)
        self.update_free_map(observations)
        # Calculate agent position in map coordinates (used throughout the act function)
        self.agent_x = int(self.full_pose[0].cpu().numpy() * 100 / self.map_resolution)
        self.agent_y = int((self.map_size_cm / 100 - self.full_pose[1].cpu().numpy()) * 100 / self.map_resolution)
        self.agent_map_pos = (self.agent_y, self.agent_x)

        if self.using_random_goal:
            self.using_random_goal = 0
            self.move_since_random += 1

        # Improved stuck detection: only count forward actions that don't move
        # Actions: 0=Stop, 1=Forward, 2=Left, 3=Right
        current_pos = observations["gps"]
        current_orientation = self.full_pose[2].cpu().numpy().item()
        position_moved = np.linalg.norm(current_pos - self.last_gps) >= 0.05
        orientation_changed = abs(current_orientation - self.last_orientation) > 5.0  # 5 degree threshold

        if position_moved:
            # Successfully moved - reset both counters
            self.move_steps += 1
            self.not_move_steps = 0
            self.forward_no_move_count = 0
            self.in_stuck_recovery = False
        else:
            # Position didn't move - always increment not_move_steps regardless of action
            self.not_move_steps += 1

            # Track consecutive failed forward actions for quick stuck detection
            if self.prev_action == 1 and not orientation_changed:
                # Forward action but didn't move and didn't turn -> likely stuck
                self.forward_no_move_count += 1
                if self.forward_no_move_count >= 2:
                    print(f"[Stuck Detection] Forward action x{self.forward_no_move_count} without movement, "
                          f"not_move_steps={self.not_move_steps}")
            elif self.prev_action in [2, 3] and not self.in_stuck_recovery:
                # Turning action - reset forward count but keep not_move_steps
                # Agent is still stuck in same position even if turning
                self.forward_no_move_count = 0

        self.last_gps = current_pos
        self.last_orientation = current_orientation
        
        self.scenegraph.perception()

        self.history_pose.append(self.full_pose.cpu().detach().clone())
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)
        
        if self.args.visualize:
            self.visualize(traversible)

        is_real_world_playback = bool(getattr(self.args, "real_world_playback", False))

        if not is_real_world_playback:
            if self.total_steps == 1:
                self.sem_map_module.set_view_angles(30)
                self.free_map_module.set_view_angles(30)
                return {"action": 5}
            elif self.total_steps <= 7:
                return {"action": 6}
            elif self.total_steps == 8:
                self.sem_map_module.set_view_angles(60)
                self.free_map_module.set_view_angles(60)
                return {"action": 5}
            elif self.total_steps <= 14:
                return {"action": 6}
            elif self.total_steps <= 15:
                self.sem_map_module.set_view_angles(30)
                self.free_map_module.set_view_angles(30)
                return {"action": 4}
            elif self.total_steps <= 16:
                self.sem_map_module.set_view_angles(0)
                self.free_map_module.set_view_angles(0)
                return {"action": 4}
        if (not is_real_world_playback) and self.total_steps <= 22 and not self.found_goal:
            self.not_move_steps = 0
            self.forward_no_move_count = 0
            self.panoramic.append(observations["rgb"][:,:,[2,1,0]])
            self.panoramic_depth.append(observations["depth"])
            self.detect_objects(observations)

            # Room detection
            if DETECTOR_TYPE == 'glip':
                room_detection_result = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], rooms_captions)
            elif DETECTOR_TYPE in ['fastsam_clip', 'fastsam_text']:
                # Use FastSAM for room detection (both CLIP and text_prompt modes)
                image_rgb = observations["rgb"][:,:,[2,1,0]]
                # Split rooms_captions: "bedroom. living room. ..." -> ['bedroom', 'living room', ...]
                room_categories = [r.strip() for r in rooms_captions.rstrip('.').split('. ')]
                room_results = self.detector.detect(image_rgb, room_categories)

                # Convert to GLIP-like format
                class MockPredictions:
                    def __init__(self, bboxes, labels, scores):
                        self.bbox = torch.tensor(bboxes) if not isinstance(bboxes, torch.Tensor) else bboxes
                        # Convert labels to tensor-like object if it's a list
                        if isinstance(labels, list):
                            # Create a wrapper that has .tolist() method
                            class LabelWrapper:
                                def __init__(self, label_list):
                                    self.labels = label_list
                                def tolist(self):
                                    return self.labels
                                def __len__(self):
                                    return len(self.labels)
                                def __getitem__(self, idx):
                                    return self.labels[idx]
                            labels = LabelWrapper(labels)
                        self._fields = {'labels': labels, 'scores': torch.tensor(scores) if not isinstance(scores, torch.Tensor) else scores}
                    def get_field(self, name):
                        return self._fields.get(name)

                room_detection_result = MockPredictions(
                    room_results['bboxes'],
                    room_results['labels'],
                    room_results['scores']
                )

            self.update_room_map(observations, room_detection_result)
            if not self.found_goal:
                return {"action": 6}
        
        if self.found_goal:
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])

            col_raw = self.map_size_cm / 10 + self.goal_gps[0] * 100 / self.resolution
            row_raw = self.map_size_cm / 10 + self.goal_gps[1] * 100 / self.resolution
            col_clipped = max(0, min(self.map_size - 1, int(col_raw)))
            row_clipped = max(0, min(self.map_size - 1, int(row_raw)))

            self.goal_map[row_clipped, col_clipped] = 1

            stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)

        elif self.found_possible_goal:
            # 优先靠近可能的目标，中断当前的landmark导航
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])

            col_raw = self.map_size_cm / 10 + self.possible_goal_temp_gps[0] * 100 / self.resolution
            row_raw = self.map_size_cm / 10 + self.possible_goal_temp_gps[1] * 100 / self.resolution
            col_clipped = max(0, min(self.map_size - 1, int(col_raw)))
            row_clipped = max(0, min(self.map_size - 1, int(row_raw)))

            row_clipped, col_clipped = self.snap_goal_to_traversible(row_clipped, col_clipped, traversible)
            self.goal_map[row_clipped, col_clipped] = 1

            # 清除landmark导航状态，优先靠近可能的goal
            self.landmark_navigation_steps = 0
            print(f"[Goal Priority] Interrupting landmark navigation, approaching possible goal instead")

            stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)

        elif (np.sum(self.goal_map) == 0 and not self.found_goal and not self.found_possible_goal) or \
        (self.using_random_goal and self.move_since_random > 15 and not self.found_goal and not self.found_possible_goal):
            # Clear previous random goal area if expired
            if self.using_random_goal and self.move_since_random > 15:
                goal_x, goal_y = np.where(self.goal_map == 1)
                if len(goal_x) > 0:
                    x_0 = max(goal_x[0] - 8, 0)
                    y_0 = max(goal_y[0] - 8, 0)
                    x_1 = min(goal_x[0] + 8, self.map_size)
                    y_1 = min(goal_y[0] + 8, self.map_size)
                    self.fbe_free_map[x_0:x_1, y_0:y_1] = 0

            # Call FBE to get landmark goal
            self.goal_loc, self.landmark_idx = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])

            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                if not self.using_random_goal:
                    self.random_nav_start_step = self.total_steps
                    print(f"[Random Nav] Started at step {self.navigate_steps}")
                    self.explanation = "Random navigation: No frontier available, exploring randomly"
                else:
                    self.explanation = f"Random navigation: Continued random exploration (step {self.total_steps - self.random_nav_start_step + 1})"
                self.using_random_goal = True
            else:
                # Got landmark goal, set it on map
                self.fronter_this_ex += 1
                self.goal_map[int(self.goal_loc[0]), int(self.goal_loc[1])] = 1
                self.goal_map = self.goal_map[::-1]

        else:
            stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)

        # If found possible goal but can't reach it, clear it
        if self.found_possible_goal and number_action == 0:
            self.found_possible_goal = False

        self.loop_time = 0
        max_recovery_attempts = 20

        while (not self.found_goal and number_action == 0) or self.forward_no_move_count >= 3 or self.not_move_steps >= 12 or self.landmark_navigation_steps >= self.max_landmark_navigation_steps:
            self.loop_time += 1
            if self.loop_time > max_recovery_attempts:
                self.loop_time = 0
                self.random_this_ex += 1

                # Mark current landmark as unreachable if we were navigating to one
                if not self.using_random_goal and self.goal_loc is not None:
                    landmark_pos = (int(self.goal_loc[0]), int(self.goal_loc[1]))
                    self.unreachable_landmarks.add(landmark_pos)
                    print(f"[Unreachable Landmark] Marked landmark at {landmark_pos} as unreachable (max recovery attempts)")

                self.goal_map = self.set_random_goal()
                if not self.using_random_goal:
                    self.random_nav_start_step = self.total_steps
                    print(f"[Random Nav] Started at step {self.navigate_steps}")
                    self.explanation = "Random navigation: No frontier available, exploring randomly"
                else:
                    self.explanation = f"Random navigation: Continued random exploration (step {self.total_steps - self.random_nav_start_step + 1})"
                self.using_random_goal = True
                # Exit recovery loop after max attempts to prevent infinite loop
                break

            # Case 1: Reached landmark but no goal found -> Call FBE
            if not self.found_goal and number_action == 0 or self.landmark_navigation_steps >= self.max_landmark_navigation_steps:
                self.not_move_steps = 0
                self.landmark_navigation_steps = 0

                self.goal_loc, self.landmark_idx = self.fbe(traversible, cur_start)
                self.not_use_random_goal()
                self.goal_map = np.zeros(self.full_map.shape[-2:])

                if self.goal_loc is None:
                    self.random_this_ex += 1
                    self.goal_map = self.set_random_goal()
                    if not self.using_random_goal:
                        self.random_nav_start_step = self.total_steps
                        print(f"[Random Nav] Started at step {self.navigate_steps}")
                        self.explanation = "Random navigation: No frontier available, exploring randomly"
                    else:
                        self.explanation = f"Random navigation: Continued random exploration (step {self.navigate_steps - self.random_nav_start_step + 1})"
                    self.using_random_goal = True
                else:
                    self.fronter_this_ex += 1
                    self.goal_map[int(self.goal_loc[0]), int(self.goal_loc[1])] = 1
                    self.goal_map = self.goal_map[::-1]


            # Case 2: Agent stuck (3 consecutive forward failures or 8+ steps without position change)
            elif self.forward_no_move_count >= 3 or self.not_move_steps >= 12:
                self.reset_goal_detection()  # Reset all goal detection state when stuck

                # If stuck for too long (>15 steps), directly use random navigation
                if self.not_move_steps > 12:
                    print(f"[Stuck Recovery] Agent stuck for {self.not_move_steps} steps (>20), using random navigation directly")
                    self.not_move_steps = 0
                    self.random_this_ex += 1

                    # Mark current landmark as unreachable if we were navigating to one
                    if not self.using_random_goal and self.goal_loc is not None:
                        landmark_pos = (int(self.goal_loc[0]), int(self.goal_loc[1]))
                        self.unreachable_landmarks.add(landmark_pos)
                        print(f"[Unreachable Landmark] Marked landmark at {landmark_pos} as unreachable (stuck)")

                    self.goal_map = self.set_random_goal()
                    self.explanation = f"Stuck for {self.not_move_steps} steps, using random navigation"

                    if not self.using_random_goal:
                        self.random_nav_start_step = self.total_steps
                        print(f"[Random Nav] Started at step {self.navigate_steps}")
                    self.using_random_goal = True

                    # Reset stuck recovery state
                    self.in_stuck_recovery = False
                    self.stuck_recovery_target_angle = None
                    self.stuck_recovery_phase = None
                    break

                else:
                    # Try stuck recovery first
                    print(f"\n[Stuck Recovery] Agent stuck for {self.not_move_steps} steps, initiating recovery...")

                    frames_back = 4
                    if len(self.orientation_history) >= frames_back:
                        stuck_orientation = self.orientation_history[-frames_back]
                        print(f"[Stuck Recovery] Using orientation from {frames_back} frames ago: {stuck_orientation:.1f}°")
                    elif len(self.orientation_history) > 0:
                        stuck_orientation = self.orientation_history[0]
                        print(f"[Stuck Recovery] Using earliest orientation (only {len(self.orientation_history)} frames): {stuck_orientation:.1f}°")
                    else:
                        # Fallback to current orientation if no history
                        stuck_orientation = self.full_pose[2].cpu().numpy().item()
                        print(f"[Stuck Recovery] No orientation history, using current: {stuck_orientation:.1f}°")

                    # Try to find an open direction using sector-based detection
                    _, _, target_angle = self.navigation_planner.find_open_direction_for_stuck_recovery(
                        self.agent_map_pos, stuck_orientation
                    )


                    # Found open direction (target_angle is already absolute and normalized)
                    # Store target angle and start turning phase
                    self.stuck_recovery_target_angle = target_angle
                    self.stuck_recovery_phase = "turning"
                    self.in_stuck_recovery = True

                    # Calculate relative angle for display
                    relative_angle = target_angle - self.full_pose[2].cpu().numpy().item()
                    while relative_angle > 180:
                        relative_angle -= 360
                    while relative_angle < -180:
                        relative_angle += 360

                    direction_str = f"Turn {'right' if relative_angle > 0 else 'left'} {abs(relative_angle):.0f}°" if abs(relative_angle) > 5 else "Forward"
                    self.explanation = f"Stuck detected ({self.not_move_steps} steps)! → Target direction: {target_angle:.1f}° ({direction_str})"
                    print(f"[Stuck Recovery] Target angle: {target_angle:.1f}° (stuck at {stuck_orientation:.1f}°)")
                    print(f"[Stuck Recovery] Phase: turning → forward → resume")

                    # End any ongoing random navigation
                    self.not_use_random_goal()

            # Handle stuck recovery action sequence (before planning)
            if self.in_stuck_recovery and self.stuck_recovery_phase is not None:
                current_orientation = self.full_pose[2].cpu().numpy().item()

                if self.stuck_recovery_phase == "turning":
                    # Calculate angle difference to target
                    angle_diff = self.stuck_recovery_target_angle - current_orientation

                    # Normalize angle difference to (-180, 180)
                    while angle_diff > 180:
                        angle_diff -= 360
                    while angle_diff < -180:
                        angle_diff += 360

                    # Check if aligned (within ±15° threshold)
                    if abs(angle_diff) <= 15:
                        # Aligned! Switch to forward phase
                        self.stuck_recovery_phase = "forward"
                        print(f"[Stuck Recovery] Aligned to target angle {self.stuck_recovery_target_angle:.1f}° (current: {current_orientation:.1f}°)")
                        print(f"[Stuck Recovery] Phase: turning → forward")
                        number_action = 1  # MOVE_FORWARD
                        
                    else:
                        # Need to turn
                        # Choose shorter turning direction
                        if angle_diff < 0:
                            # Target is counterclockwise (left) from current
                            number_action = 3  # TURN_RIGHT (in habitat coordinate system)
                            turn_dir = "right"
                            print(f"[Stuck Recovery] Turning right {abs(angle_diff):.1f}° to reach {self.stuck_recovery_target_angle:.1f}°")

                        else:
                            # Target is clockwise (right) from current
                            number_action = 2  # TURN_LEFT
                            turn_dir = "left"
                            print(f"[Stuck Recovery] Turning left {abs(angle_diff):.1f}° to reach {self.stuck_recovery_target_angle:.1f}°")

                        self.explanation = f"Stuck recovery: Turning {turn_dir} {abs(angle_diff):.1f}° → target direction: {self.stuck_recovery_target_angle:.1f}°"
                        
                elif self.stuck_recovery_phase == "forward":
                    # Execute forward step
                    number_action = 1  # MOVE_FORWARD
                    print(f"[Stuck Recovery] Executing forward step in open direction")
                    self.explanation = f"Stuck recovery: Taking forward step toward {self.stuck_recovery_target_angle:.1f}°"

                    # Reset stuck recovery state and resume normal navigation
                    self.in_stuck_recovery = False
                    self.stuck_recovery_target_angle = None
                    self.stuck_recovery_phase = None
                    print(f"[Stuck Recovery] Completed! Resuming normal navigation")
                
                break

            else:
                # Normal planning (not in stuck recovery or phase is None)
                # Replan with new goal
                stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)

        # Calculate distance to current goal for visualization
        self.distance_to_goal = self._calculate_distance_to_goal()

        if not self.using_random_goal:
            self.landmark_navigation_steps += 1

        observations["pointgoal_with_gps_compass"] = self.get_relative_goal_gps(observations)

        self.last_loc = copy.deepcopy(self.full_pose)
        self.prev_action = number_action
        self.navigate_steps += 1


        # Record current orientation to history (for stuck recovery)
        current_orientation = self.full_pose[2].cpu().numpy().item()
        print(f'navigate_steps: {self.navigate_steps}, curr ori: {int(current_orientation)}, distance_to_goal: {self.distance_to_goal}')
        self.orientation_history.append(current_orientation)
        # Keep only last N frames
        if len(self.orientation_history) > self.orientation_history_size:
            self.orientation_history.pop(0)

        torch.cuda.empty_cache()

        return {"action": number_action}
    
    def not_use_random_goal(self):
        # Record end of random navigation period
        if self.using_random_goal and self.random_nav_start_step is not None:
            self.random_nav_periods.append((self.random_nav_start_step, self.total_steps))
            print(f"[Random Nav] Ended: steps {self.random_nav_start_step}-{self.total_steps} (duration: {self.total_steps - self.random_nav_start_step} steps)")
            self.random_nav_start_step = None

        self.move_since_random = 0
        self.using_random_goal = False

    def _calculate_distance_to_goal(self):
        """
        Calculate the distance from agent to the current goal in the goal_map.

        Returns:
            Distance in meters, or None if no goal is set
        """
        if not hasattr(self, 'goal_map') or self.goal_map is None:
            return None

        # Find goal position in goal_map
        goal_positions = np.where(self.goal_map == 1)
        if len(goal_positions[0]) == 0:
            return None

        # Get first goal position (in case multiple goals)
        goal_y = goal_positions[0][0]  # row
        goal_x = goal_positions[1][0]  # col

        agent_y, agent_x = self.agent_map_pos
        # Calculate pixel distance
        pixel_distance = np.sqrt((goal_x - agent_x)**2 + (goal_y - agent_y)**2)

        # Convert to meters
        distance_meters = pixel_distance * self.map_resolution / 100.0

        return distance_meters

    def find_all_valid_frontier_midpoints(self, frontier_locations, trajectory_exclusion_radius=31,
                                           agent_position=None, min_distance_from_agent=30, min_cluster_size=2,
                                           min_linearity_ratio=0.1, fbe_free_map=None):
        """Find all valid frontier midpoints - delegates to LandmarkManager."""
        return self.landmark_manager.find_all_valid_frontier_midpoints(
            frontier_locations, trajectory_exclusion_radius, agent_position, min_distance_from_agent,
            min_cluster_size, min_linearity_ratio, fbe_free_map=fbe_free_map)

    def filter_landmarks(self, landmark_nodes, agent_position, min_distance=20, trajectory_exclusion_radius=31):
        """Filter landmarks - delegates to LandmarkManager."""
        return self.landmark_manager.filter_landmarks(landmark_nodes, agent_position, min_distance, trajectory_exclusion_radius)

    def filter_unreachable_landmarks(self, landmarks):
        """
        Filter out landmarks that are known to be unreachable or are too close to unreachable landmarks.

        Args:
            landmarks: List or array of landmarks, each as [row, col]

        Returns:
            Filtered list of landmarks (as list), with unreachable ones removed
        """
        if not landmarks or len(landmarks) == 0:
            return []

        if not self.unreachable_landmarks:
            # No unreachable landmarks yet, return all
            return landmarks if isinstance(landmarks, list) else landmarks.tolist()

        filtered_landmarks = []
        for lm in landmarks:
            lm_pos = np.array([lm[0], lm[1]])
            is_too_close = False

            # Check distance to all unreachable landmarks
            for unreachable_pos in self.unreachable_landmarks:
                unreachable_np = np.array([unreachable_pos[0], unreachable_pos[1]])
                dist = np.linalg.norm(lm_pos - unreachable_np)

                if dist <= self.unreachable_landmark_radius:
                    is_too_close = True
                    print(f"[Filter Unreachable] Excluding landmark at ({lm[0]},{lm[1]}) - too close to unreachable landmark at {unreachable_pos} (dist={dist:.1f}px)")
                    break

            if not is_too_close:
                filtered_landmarks.append(lm)

        return filtered_landmarks

    def fbe(self, traversible, start):
        room_nodes = self.scenegraph.room_nodes
        object_nodes = self.scenegraph.nodes

        frontier_map, frontier_locations, num_frontiers = calculate_frontiers(self.full_map, self.fbe_free_map)
        if frontier_locations is None or len(frontier_locations) == 0:
            return None, None
        fl_np = frontier_locations.cpu().numpy()
        self.landmark_map.update(fl_np, self.fbe_free_map, self.full_map, traversible[1:-1, 1:-1],
                                 agent_position=start, agent_trajectory_map=self.agent_trajectory_map)

        landmark_nodes = self.landmark_map.get_nodes()

        # Use agent_map_pos calculated in act() function to avoid redundant calculation
        agent_y, agent_x = self.agent_map_pos

        if landmark_nodes is not None and len(landmark_nodes) > 0:
            landmark_nodes = self.filter_landmarks(landmark_nodes, self.agent_map_pos, min_distance=30, trajectory_exclusion_radius=30)

        self.landmark_near_frontier = {}  # 存储每个 landmark 是否靠近 frontier
        frontier_proximity_threshold = 10.0  # pixels (可调整)

        if landmark_nodes is not None and len(landmark_nodes) > 0 and frontier_locations is not None:
            for i, lm in enumerate(landmark_nodes):
                row, col = int(lm[0]), int(lm[1])
                lm_pos = np.array([row, col])

                # 计算到最近 frontier 的距离
                min_dist_to_frontier = float('inf')
                for frontier_pos in fl_np:
                    # frontier_pos is [row, col]
                    dist = np.linalg.norm(lm_pos - frontier_pos)
                    if dist < min_dist_to_frontier:
                        min_dist_to_frontier = dist

                # 判断是否靠近 frontier
                is_near_frontier = min_dist_to_frontier <= frontier_proximity_threshold
                dist_to_frontier_meters = min_dist_to_frontier * self.map_resolution / 100.0

                # 使用位置作为 key 存储信息
                landmark_key = (row, col)
                self.landmark_near_frontier[landmark_key] = {
                    'near_frontier': is_near_frontier,
                    'distance_to_frontier': float(min_dist_to_frontier),
                    'distance_to_frontier_meters': float(dist_to_frontier_meters)
                }

                print(f"Landmark {i} ({self.map_height - row},{col}): {'Near' if is_near_frontier else 'Far from'} frontier, dist={dist_to_frontier_meters:.2f}m")


        # Use frontier midpoints as landmarks ONLY when graph-based landmarks are unavailable
        # This is a fallback when no Voronoi landmarks can be generated
        if landmark_nodes is None or len(landmark_nodes) == 0:
            frontier_midpoints = self.find_all_valid_frontier_midpoints(
                fl_np, trajectory_exclusion_radius=10, agent_position=self.agent_map_pos, min_distance_from_agent=26,
                fbe_free_map=self.fbe_free_map)

            if frontier_midpoints is not None and len(frontier_midpoints) > 0:
                landmark_nodes = frontier_midpoints
                self.landmarks = landmark_nodes.tolist()


        # Store initial landmarks
        if landmark_nodes is not None and len(landmark_nodes) > 0:
            self.landmarks = landmark_nodes.tolist()
        else:
            self.landmarks = []
            landmark_nodes = None

        if landmark_nodes is not None and len(landmark_nodes) > 0:
            original_count = len(landmark_nodes)
            # Convert to list if needed
            landmarks_list = landmark_nodes if isinstance(landmark_nodes, list) else landmark_nodes.tolist()

            # Apply unreachable landmark filter
            filtered_landmarks = self.filter_unreachable_landmarks(landmarks_list)

            if len(filtered_landmarks) > 0:
                landmark_nodes = np.array(filtered_landmarks)
                self.landmarks = landmark_nodes.tolist()
                if len(filtered_landmarks) < original_count:
                    print(f"[Filter Unreachable] Filtered out {original_count - len(filtered_landmarks)} unreachable landmarks, "
                          f"{len(filtered_landmarks)} remaining")
            else:
                # All landmarks filtered out
                print(f"[Filter Unreachable] All {original_count} landmarks filtered out as unreachable")
                landmark_nodes = None

        # Check if we have any landmarks after all processing
        if landmark_nodes is None or len(landmark_nodes) == 0:
            self.landmark_sg_pairs = None
            self.explanation = "No landmarks or frontiers available"
            return None, None

        enumerated_groups = EnumeratedGroupNodes()
        enumerated_groups.add_from_rooms(
            room_nodes,
            enumerate_func=self.scenegraph.enumerate_group_node,
            top_k_per_group=3,
            min_prob_threshold = 0.0
        )

        enable_llm_filter = True
        if enable_llm_filter:
            if not hasattr(self, 'llm_filter'):
                self.llm_filter = LLMFilter(verbose=False)

            if len(enumerated_groups) > 0:
                original_count = len(enumerated_groups)
                enumerated_groups.apply_llm_filter(self.llm_filter)

        self.enumerated_groups = enumerated_groups

        room_combinations = RoomGroupCombinations()
        room_combinations.add_from_filtered_enumerations(
            enumerated_groups,
            top_k=8,
            min_prob_threshold=0.0
        )

        if enable_llm_filter:
            if len(room_combinations) > 0:
                original_count = len(room_combinations)
                room_combinations.apply_llm_filter(self.llm_filter)

        self.room_combinations = room_combinations

        scenegraph_combinations = SceneGraphCombinations()
        scenegraph_combinations.add_from_room_combinations(
            room_combinations,
            max_combinations=16,
            min_prob_threshold=0.00
        )

        self.scenegraph_combinations = scenegraph_combinations
        # Case 2: Have landmarks but no scene-graph combinations → select nearest landmark
        if scenegraph_combinations is None or len(scenegraph_combinations) == 0:
            self.landmark_sg_pairs = None

            # Find the nearest landmark
            min_dist = float('inf')
            nearest_landmark = None
            nearest_landmark_idx = None
            for idx, landmark in enumerate(landmark_nodes):
                # landmark is [col, row] in pixels
                landmark_pos = np.array([landmark[0], landmark[1]])
                agent_pos = np.array([agent_x, agent_y])
                dist = np.linalg.norm(landmark_pos - agent_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_landmark = landmark
                    nearest_landmark_idx = idx

            if nearest_landmark is not None:
                self.explanation = f"No scene-graphs available. Navigating to nearest landmark at ({self.map_height - nearest_landmark[0]:.0f}, {nearest_landmark[1]:.0f})"
                goal_from_landmark = np.array(nearest_landmark) - 1
                return goal_from_landmark, nearest_landmark_idx
            else:
                return None, None

        goal_from_landmark, landmark_idx, explanation = \
            self.landmark_selector.select_best_landmark_by_comparison(
                landmark_nodes=landmark_nodes,
                scenegraph_combinations=scenegraph_combinations,
                object_nodes=object_nodes,
                agent_map_pos=self.agent_map_pos,
                obj_goal_sg=self.obj_goal_sg,
            )

        if goal_from_landmark is None:
            print(f"[FBE] All landmarks filtered out by comparison selector, trying frontier midpoints fallback...")
            frontier_midpoints = self.find_all_valid_frontier_midpoints(
                fl_np, trajectory_exclusion_radius=21, agent_position=self.agent_map_pos, min_distance_from_agent=21,
                fbe_free_map=self.fbe_free_map)

            if frontier_midpoints is not None and len(frontier_midpoints) > 0:
                landmark_nodes = frontier_midpoints
                self.landmarks = landmark_nodes.tolist()
                min_dist = float('inf')
                nearest_frontier = None
                nearest_frontier_idx = None
                for idx, frontier in enumerate(frontier_midpoints):
                    frontier_pos = np.array([self.map_height - frontier[0], frontier[1]])
                    dist = np.linalg.norm(frontier_pos - self.agent_map_pos)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_frontier = frontier
                        nearest_frontier_idx = idx

                if nearest_frontier is not None:
                    self.explanation = f"All Voronoi landmarks filtered out. Using nearest frontier midpoint at ({self.map_height - nearest_frontier[0]:.0f}, {nearest_frontier[1]:.0f})"
                    goal_from_landmark = np.array(nearest_frontier) - 1
                    return goal_from_landmark, nearest_frontier_idx

        self.explanation = explanation

        return goal_from_landmark, landmark_idx

    def get_unoccluded_center_point(self, goal_box, goal_bbox_idx, depth_map):
        """
        Calculate the center point of goal bbox and its true depth using unoccluded regions.

        Args:
            goal_box: Goal bbox tensor [x1, y1, x2, y2]
            goal_bbox_idx: Index of goal in current_obj_predictions
            depth_map: Depth map array

        Returns:
            tuple: (center_point, is_occluded, true_depth) where
                   - center_point is [x, y]
                   - is_occluded is boolean
                   - true_depth is the average depth of unoccluded center 80% region
        """
        goal_box = goal_box.to(torch.int64)
        x1, y1, x2, y2 = goal_box[0].item(), goal_box[1].item(), goal_box[2].item(), goal_box[3].item()

        # Calculate center 80% region of the bbox
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * 0.1)  # 10% margin on each side
        margin_y = int(height * 0.1)  # 10% margin on top and bottom

        center80_x1 = x1 + margin_x
        center80_y1 = y1 + margin_y
        center80_x2 = x2 - margin_x
        center80_y2 = y2 - margin_y

        # Find overlapping objects that might occlude the goal
        all_bboxes = self.current_obj_predictions.bbox
        occluding_regions = []

        # First pass: sample depth to get a rough estimate for occlusion detection
        sample_points = []
        for dy in [0.3, 0.5, 0.7]:
            for dx in [0.3, 0.5, 0.7]:
                px = int(x1 + (x2 - x1) * dx)
                py = int(y1 + (y2 - y1) * dy)
                if 0 <= py < depth_map.shape[0] and 0 <= px < depth_map.shape[1]:
                    depth_val = depth_map[py, px, 0]
                    if depth_val < 100:  # Valid depth
                        sample_points.append(depth_val)

        # Use median depth as rough estimate for occlusion detection
        rough_depth = np.median(sample_points) if len(sample_points) > 0 else depth_map[(y1+y2)//2, (x1+x2)//2, 0]

        # Find occluding regions
        for idx, other_box in enumerate(all_bboxes):
            if idx == goal_bbox_idx:
                continue  # Skip self

            other_box = other_box.to(torch.int64)
            ox1, oy1, ox2, oy2 = other_box[0].item(), other_box[1].item(), other_box[2].item(), other_box[3].item()

            # Check if bboxes overlap with center 80% region
            overlap_x1 = max(center80_x1, ox1)
            overlap_y1 = max(center80_y1, oy1)
            overlap_x2 = min(center80_x2, ox2)
            overlap_y2 = min(center80_y2, oy2)

            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                # Bboxes overlap, check depth to determine occlusion
                ocx = (overlap_x1 + overlap_x2) // 2
                ocy = (overlap_y1 + overlap_y2) // 2
                if 0 <= ocy < depth_map.shape[0] and 0 <= ocx < depth_map.shape[1]:
                    other_depth = depth_map[ocy, ocx, 0]

                    # If other object is closer (smaller depth), it's occluding the goal
                    if other_depth < rough_depth and other_depth < 100:
                        occluding_regions.append([overlap_x1, overlap_y1, overlap_x2, overlap_y2])

        # Calculate true depth from unoccluded center 80% region
        is_occluded = len(occluding_regions) > 0

        # Collect depth values from unoccluded pixels in center 80% region
        unoccluded_depths = []

        for py in range(center80_y1, center80_y2):
            for px in range(center80_x1, center80_x2):
                if not (0 <= py < depth_map.shape[0] and 0 <= px < depth_map.shape[1]):
                    continue

                # Check if this pixel is occluded
                is_pixel_occluded = False
                for occ_region in occluding_regions:
                    if (occ_region[0] <= px <= occ_region[2] and
                        occ_region[1] <= py <= occ_region[3]):
                        is_pixel_occluded = True
                        break

                if not is_pixel_occluded:
                    depth_val = depth_map[py, px, 0]
                    if depth_val < 100:  # Valid depth
                        unoccluded_depths.append(depth_val)

        # Calculate true depth as mean of unoccluded depths
        if len(unoccluded_depths) > 0:
            true_depth = np.mean(unoccluded_depths)
        else:
            # Fallback: use rough depth if no valid unoccluded pixels
            true_depth = rough_depth
            print(f"[Warning] No unoccluded pixels in center 80% region, using rough depth: {true_depth:.2f}")

        # Center point calculation
        if not is_occluded:
            # No occlusion, use simple center
            center_point = torch.tensor([(x1 + x2) // 2, (y1 + y2) // 2])
        else:
            # Find the centroid of unoccluded region
            unoccluded_pixels = []
            for py in range(center80_y1, center80_y2):
                for px in range(center80_x1, center80_x2):
                    if not (0 <= py < depth_map.shape[0] and 0 <= px < depth_map.shape[1]):
                        continue

                    # Check if this pixel is occluded
                    is_pixel_occluded = False
                    for occ_region in occluding_regions:
                        if (occ_region[0] <= px <= occ_region[2] and
                            occ_region[1] <= py <= occ_region[3]):
                            is_pixel_occluded = True
                            break

                    if not is_pixel_occluded:
                        unoccluded_pixels.append([px, py])

            if len(unoccluded_pixels) > 0:
                # Use centroid of unoccluded pixels
                unoccluded_pixels = np.array(unoccluded_pixels)
                center_x = int(np.mean(unoccluded_pixels[:, 0]))
                center_y = int(np.mean(unoccluded_pixels[:, 1]))
                center_point = torch.tensor([center_x, center_y])
                print(f"[Occlusion] Using centroid of unoccluded region: ({center_x}, {center_y}), "
                      f"{len(unoccluded_pixels)} unoccluded pixels, true_depth: {true_depth:.2f}")
            else:
                # Fallback: use bbox center
                center_point = torch.tensor([(x1 + x2) // 2, (y1 + y2) // 2])
                print(f"[Warning] No unoccluded pixels, using bbox center")

        return center_point, is_occluded, true_depth


    def get_goal_gps(self, observations, angle, distance):
        """Calculate goal GPS - delegates to NavigationPlanner."""
        return self.navigation_planner.get_goal_gps(observations, angle, distance)

    def get_relative_goal_gps(self, observations, goal_gps=None):
        """Calculate relative goal GPS - delegates to NavigationPlanner."""
        return self.navigation_planner.get_relative_goal_gps(observations, goal_gps)

    def init_map(self):
        """Initialize maps - delegates to MapManager."""
        return self.map_manager.init_map()

    def update_map(self, observations):
        """Update obstacle map - delegates to MapManager."""
        return self.map_manager.update_map(observations)
    
    def update_free_map(self, observations):
        """Update free space map - delegates to MapManager."""
        return self.map_manager.update_free_map(observations)
    
    def update_room_map(self, observations, room_prediction_result):
        """Update room map - delegates to MapManager."""
        return self.map_manager.update_room_map(observations, room_prediction_result)
    
    def get_traversible(self, map_pred, pose_pred):
        """Get traversible area - delegates to MapManager."""
        return self.map_manager.get_traversible(map_pred, pose_pred)

    def _plan(self, traversible, goal_map, agent_pose, start, start_o, goal_found):
        if self.prev_action == 1:
            x1, y1, t1 = self.last_loc.cpu().numpy()
            x2, y2, t2 = self.full_pose.cpu()
            y1 = self.map_size_cm/100 - y1
            y2 = self.map_size_cm/100 - y2
            t1 = -t1
            t2 = -t2
            buf = 4
            length = 5

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            col_threshold = self.collision_threshold

            if dist < col_threshold: # Collision
                self.former_collide += 1
                for i in range(length):
                    wx = x1 + 0.05 * ((i + buf) * np.cos(np.deg2rad(t1)))
                    wy = y1 + 0.05 * ((i + buf) * np.sin(np.deg2rad(t1)))
                    r, c = wy, wx
                    r = int(round(r * 100 / self.map_resolution))
                    c = int(round(c * 100 / self.map_resolution))
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r,c] = 1
            else:
                self.former_collide = 0

        stg, replan, stop, = self._get_stg(traversible, start, np.copy(goal_map), goal_found)

        # Deterministic Local Policy
        if stop:
            action = 0
            (stg_y, stg_x) = stg

        else:
            (stg_y, stg_x) = stg
            angle_st_goal = math.degrees(math.atan2(stg_y - start[0],
                                                stg_x - start[1]))
            angle_agent = (start_o)%360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_st_goal- angle_agent)%360.0
            if relative_angle > 180:
                relative_angle -= 360
            if self.former_collide < 10:
                if relative_angle > 16:
                    action = 3 # Right
                elif relative_angle < -16:
                    action = 2 # Left
                else:
                    action = 1
            elif self.prev_action == 1:
                if relative_angle > 0:
                    action = 3 # Right
                else:
                    action = 2 # Left
            else:
                action = 1
            if self.former_collide >= 10 and self.prev_action != 1:
                self.former_collide  = 0
            if stg_y == start[0] and stg_x == start[1]:
                action = 1

        return stg_y, stg_x, replan, action
    
    def _get_stg(self, traversible, start, goal, goal_found):
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        goal = add_boundary(goal, value=0)
        original_goal = copy.deepcopy(goal)
        
        centers = []
        if len(np.where(goal !=0)[0]) > 1:
            goal, centers = CH._get_center_goal(goal)
        state = [start[0] + 1, start[1] + 1]
        self.planner = FMMPlanner(traversible, None)
            
        if self.dilation_deg!=0: 
            goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
            
        if goal_found:
            try:
                goal = CH._block_goal(centers, goal, original_goal, goal_found)
            except:
                goal = self.set_random_goal(goal)

        self.planner.set_multi_goal(goal, state) # time cosuming

        decrease_stop_cond = -0.5
        stg_y, stg_x, replan, stop = self.planner.get_short_term_goal(state, found_goal = goal_found, decrease_stop_cond=decrease_stop_cond)
        stg_x, stg_y = stg_x - 1, stg_y - 1
        
        return (stg_y, stg_x), replan, stop

    def set_random_goal(self):
        obstacle_map = self.full_map.cpu().numpy()[0,0,::-1]
        goal = np.zeros_like(obstacle_map)
        goal_index = np.where((obstacle_map<1))
        np.random.seed(self.total_steps)
        if len(goal_index[0]) != 0:
            i = np.random.choice(len(goal_index[0]), 1)[0]
            h_goal = goal_index[0][i]
            w_goal = goal_index[1][i]
        else:
            h_goal = np.random.choice(goal.shape[0], 1)[0]
            w_goal = np.random.choice(goal.shape[1], 1)[0]
        goal[h_goal, w_goal] = 1
        return goal

    def update_metrics(self, metrics):
        self.metrics['distance_to_goal'] = metrics['distance_to_goal']
        self.metrics['spl'] = metrics['spl']
        self.metrics['softspl'] = metrics['softspl']
        # Save success metric for RAG decision making
        if 'success' in metrics:
            self.metrics['success'] = metrics['success']

        # Print random navigation summary at episode end
        if self.simulator._env.episode_over or self.total_steps == 500:
            # Close any ongoing random navigation period
            if self.using_random_goal and self.random_nav_start_step is not None:
                self.random_nav_periods.append((self.random_nav_start_step, self.total_steps))

            episode_success = self.metrics.get('success', 0.0)

            # Only collect if rag_collect_enabled=True
            if self.rag_collect_enabled and self.rag_manager is not None:
                if episode_success == 1.0:
                    # Prefer rag_latest_goal_detection (most recent confirmed detection)
                    # Fallback to latest detection from sliding window if not available
                    if self.rag_latest_goal_detection is not None:
                        detection_to_save = self.rag_latest_goal_detection
                        source = "latest_goal_detection"
                    elif len(self.rag_sliding_window) > 0:
                        # Use most recent detection (not highest confidence!)
                        detection_to_save = self.rag_sliding_window[-1]
                        source = "sliding_window[-1]"
                    else:
                        detection_to_save = None
                        source = None

                    if detection_to_save is not None:
                        self.rag_manager.add_successful_detection(
                            obj_category=self.obj_goal,
                            crop=detection_to_save['crop'],
                            caption=detection_to_save['caption'],
                            confidence=detection_to_save['confidence'],
                            episode_id=str(self.count_episodes),
                            dataset=self.current_dataset,
                            scene_id=self.current_scene_id
                        )

                        print(f"\n{'='*60}")
                        print(f"{'='*60}")
                        print(f"Added to knowledge base:")
                        print(f"  - Goal: {self.obj_goal}")
                        print(f"  - Caption: {detection_to_save['caption']}")
                        print(f"  - Confidence: {detection_to_save['confidence']:.3f}")
                        print(f"  - Dataset: {self.current_dataset}, Scene: {self.current_scene_id}")
                        print(f"  - Source: {source}")
                        if source == "sliding_window[-1]":
                            print(f"  - Total detections in window: {len(self.rag_sliding_window)}")
                        print(f"{'='*60}\n")
                elif episode_success == 0.0 and len(self.rag_sliding_window) > 0:
                    # Episode failed - add detections to false positive database
                    # Add all detections from sliding window as potential false positives
                    print(f"\n{'='*60}")
                    print(f"❌ [RAG] Episode {self.count_episodes} FAILED")
                    print(f"{'='*60}")
                    print(f"Adding {len(self.rag_sliding_window)} detection(s) to false positive database:")
                    print(f"  - Dataset: {self.current_dataset}, Scene: {self.current_scene_id}")

                    for detection in self.rag_sliding_window:
                        self.rag_manager.add_false_positive(
                            obj_category=self.obj_goal,
                            crop=detection['crop'],
                            caption=detection['caption'],
                            confidence=detection['confidence'],
                            episode_id=str(self.count_episodes),
                            dataset=self.current_dataset,
                            scene_id=self.current_scene_id
                        )
                        print(f"  - Caption: {detection['caption']}, Confidence: {detection['confidence']:.3f}")

                    print(f"{'='*60}\n")
                elif episode_success == 0.0:
                    print(f"\n{'='*60}")
                    print(f"{'='*60}")
                    print(f"Not adding to false positive database")
                    print(f"  - Window was empty (no detections)")
                    print(f"{'='*60}\n")
            elif not self.rag_collect_enabled:
                # RAG collection disabled
                if episode_success == 1.0:
                    print(f"[RAG] Episode {self.count_episodes} SUCCESS but collection disabled (rag_mode={self.rag_mode})")

            print(f"\n{'='*60}")
            print(f"Random Navigation Summary - Episode {self.count_episodes}")
            print(f"{'='*60}")
            if len(self.random_nav_periods) > 0:
                total_random_steps = sum(end - start for start, end in self.random_nav_periods)
                print(f"Total episodes: {len(self.random_nav_periods)}")
                print(f"Total steps in random nav: {total_random_steps}/{self.total_steps} ({100*total_random_steps/self.total_steps:.1f}%)")
                print(f"\nDetailed periods:")
                for i, (start, end) in enumerate(self.random_nav_periods, 1):
                    print(f"  Period {i}: steps {start:3d}-{end:3d} (duration: {end-start:3d} steps)")
            else:
                print("No random navigation used in this episode")
            print(f"{'='*60}\n")

        if self.args.visualize:
            if self.simulator._env.episode_over or self.total_steps == 500:
                # delegate to visualizations module
                from visualizations import save_video as _save_video
                _save_video(self)

    def visualize(self, traversible):
        return _visualize(self, traversible)
    
        

    def save_video(self):
        # delegating implementation to visualizations module
        from visualizations import save_video as _save_video
        return _save_video(self)


    def visualize_agent_and_goal(self, map):
        from visualizations import visualize_agent_and_goal as _vag
        return _vag(self, map)

    def _save_room_map_with_grid(self, room_map_np, room_names):
        """Save room map visualization - delegates to MapManager."""
        return self.map_manager.save_room_map_with_grid(room_map_np, room_names)


def main():
    def build_runtime_config_file(base_config_file, dataset_split):
        base_path = Path(base_config_file)
        runtime_dir = Path("/tmp/psgnav_runtime_configs")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        generated_path = runtime_dir / f"{base_path.stem}.{dataset_split}{base_path.suffix}"

        lines = base_path.read_text().splitlines()
        in_dataset_block = False
        replaced = False
        new_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped == 'DATASET:':
                in_dataset_block = True
                new_lines.append(line)
                continue

            if in_dataset_block and stripped and not line.startswith('  '):
                in_dataset_block = False

            if in_dataset_block and stripped.startswith('SPLIT:'):
                indent = line[: len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}SPLIT: {dataset_split}")
                replaced = True
                continue

            new_lines.append(line)

        if not replaced:
            raise ValueError(f"Failed to inject DATASET.SPLIT into config: {base_config_file}")

        generated_path.write_text("\n".join(new_lines) + "\n")
        return str(generated_path)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize", default=True, action='store_true'
    )
    parser.add_argument(
        "--split_l", default=0, type=int
    )
    parser.add_argument(
        "--split_r", default=20, type=int
    )
    parser.add_argument(
        "--server_port", default=None, type=int
    )
    # RAG hyperparameters
    parser.add_argument(
        "--rag_mode", default=3, type=int,
        choices=[0, 1, 2, 3],
        help="RAG mode: 0=disabled, 1=collect_only, 2=use_only, 3=full (use+collect, default)"
    )
    parser.add_argument(
        "--rag_max_docs", default=10, type=int,
        help="Maximum number of documents per category in RAG knowledge base"
    )
    parser.add_argument(
        "--rag_similarity_threshold", default=0.1, type=float,
        help="Similarity threshold for RAG verification (0.0-1.0)"
    )
    parser.add_argument(
        "--rag_caption_penalty", default=1.0, type=float,
        help="Penalty weight for caption mismatch in RAG (0.0-1.0)"
    )
    parser.add_argument(
        "--rag_window_size", default=5, type=int,
        help="Sliding window size for storing recent detections"
    )
    parser.add_argument(
        "--dataset", default="hssd", type=str,
        choices=list(SUPPORTED_DATASETS),
        help="Dataset to use: hssd"
    )
    parser.add_argument(
        "--dataset_split", default=None, type=str,
        help="Dataset split override. Defaults to val for HSSD."
    )
    parser.add_argument(
        "--timestamp", default=None, type=str,
        help="Timestamp for this run (format: MMDD_HHMM). If not provided, will be auto-generated."
    )
    # Episode range control
    parser.add_argument(
        "--episode_start", default=162, type=int,
        help="Starting episode index within each scene (default: 0)"
    )
    parser.add_argument(
        "--episode_end", default=None, type=int,
        help="Ending episode index within each scene (None means all episodes, default: None)"
    )
    args = parser.parse_args()

    # Select config file based on dataset argument
    base_config_file = get_dataset_config_path(args.dataset)
    dataset_split = args.dataset_split or get_default_dataset_split(args.dataset)
    config_file = build_runtime_config_file(base_config_file, dataset_split)

    os.environ["CHALLENGE_CONFIG_FILE"] = config_file
    print(f"Using dataset: {args.dataset.upper()}")
    print(f"Dataset split: {dataset_split}")
    print(f"Base config file: {base_config_file}")
    print(f"Runtime config file: {config_file}")
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    agent = PSG_Nav_Agent(task_config=config, args=args)


    challenge = habitat.Challenge(
        eval_remote=False,
        split_l=args.split_l,
        split_r=args.split_r,
        episode_start=args.episode_start,
        episode_end=args.episode_end
    )

    challenge.submit(agent)


if __name__ == "__main__":
    main()
