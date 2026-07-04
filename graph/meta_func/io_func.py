
from omegaconf import DictConfig
import omegaconf
from pathlib import Path, PosixPath
import numpy as np

class IOFunc():
    def __init__(self):   
        pass

    def set_cfg(self):
        cfg = {'dataset_config': PosixPath('tools/replica.yaml'), 'scene_id': 'room0', 'start': 0, 'end': -1, 'stride': 5, 'image_height': 680, 'image_width': 1200, 'gsa_variant': 'none', 'detection_folder_name': 'gsa_detections_${gsa_variant}', 'det_vis_folder_name': 'gsa_vis_${gsa_variant}', 'color_file_name': 'gsa_classes_${gsa_variant}', 'device': 'cuda', 'use_iou': True, 'spatial_sim_type': 'overlap', 'phys_bias': 0.0, 'match_method': 'sim_sum', 'semantic_threshold': 0.5, 'physical_threshold': 0.5, 'sim_threshold': 1.2, 'use_contain_number': False, 'contain_area_thresh': 0.95, 'contain_mismatch_penalty': 0.5, 'mask_area_threshold': 25, 'mask_conf_threshold': 0.95, 'max_bbox_area_ratio': 0.5, 'skip_bg': True, 'min_points_threshold': 16, 'downsample_voxel_size': 0.025, 'dbscan_remove_noise': True, 'dbscan_eps': 0.1, 'dbscan_min_points': 10, 'obj_min_points': 0, 'obj_min_detections': 3, 'merge_overlap_thresh': 0.7, 'merge_visual_sim_thresh': 0.8, 'merge_text_sim_thresh': 0.8, 'denoise_interval': 20, 'filter_interval': -1, 'merge_interval': 20, 'save_pcd': True, 'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub', 'vis_render': False, 'debug_render': False, 'class_agnostic': True, 'save_objects_all_frames': True, 'render_camera_path': 'replica_room0.json', 'max_num_points': 512}
        cfg = DictConfig(cfg)
        if self.is_navigation:
            cfg.sim_threshold = 0.8
            cfg.sim_threshold_spatial = 0.01
        self.cfg = cfg
    
    def process_cfg(self, cfg: DictConfig):
        cfg.dataset_root = Path(cfg.dataset_root)
        cfg.dataset_config = Path(cfg.dataset_config)
        
        if cfg.dataset_config.name != "multiscan.yaml":
            # For datasets whose depth and RGB have the same resolution
            # Set the desired image heights and width from the dataset config
            dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
            if cfg.image_height is None:
                cfg.image_height = dataset_cfg.camera_params.image_height
            if cfg.image_width is None:
                cfg.image_width = dataset_cfg.camera_params.image_width
            print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
        else:
            # For dataset whose depth and RGB have different resolutions
            assert cfg.image_height is not None and cfg.image_width is not None, \
                "For multiscan dataset, image height and width must be specified"

        return cfg

    def set_agent(self, agent):
        self.agent = agent

    def set_obj_goal(self, obj_goal, obj_goal_sg):
        self.obj_goal = obj_goal
        self.obj_goal_sg = obj_goal_sg
        if getattr(self, 'dataset', None) == 'hm3d_ovon' and obj_goal_sg:
            current_terms = [term.strip() for term in self.node_space.split('.') if term.strip()]
            if obj_goal_sg not in current_terms:
                current_terms.append(obj_goal_sg)
                self.node_space = '. '.join(current_terms) + '.'
        if self.obj_goal in self.threshold_list:
            self.cfg.obj_min_detections = self.threshold_list[self.obj_goal]

    def set_navigate_steps(self, navigate_steps):
        self.navigate_steps = navigate_steps

    def set_room_map(self, room_map):
        self.room_map = room_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map
    
    def set_observations(self, observations):
        self.observations = observations
        self.image_rgb = observations['rgb'].copy()
        self.image_depth = observations['depth'].copy()
        self.pose_matrix = self.get_pose_matrix()

    def set_frontier_map(self, frontier_map):
        self.frontier_map = frontier_map

    def set_full_map(self, full_map):
        self.full_map = full_map

    def set_fbe_free_map(self, fbe_free_map):
        self.fbe_free_map = fbe_free_map

    def set_full_pose(self, full_pose):
        self.full_pose = full_pose

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        edges = set()
        for node in self.nodes:
            edges.update(node.edges)
        edges = list(edges)
        return edges

    def get_seg_xyxy(self):
        return self.seg_xyxy

    def get_seg_caption(self):
        return self.seg_caption
    
    def get_pose_matrix(self):
        x = self.map_size_cm / 100.0 / 2.0 + self.observations['gps'][0]
        y = self.map_size_cm / 100.0 / 2.0 - self.observations['gps'][1]
        t = (self.observations['compass'] - np.pi / 2)[0] # input degrees and meters
        pose_matrix = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t), np.cos(t), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return pose_matrix
