import torch
from utils.utils_sg_uq.graph_utils import MapObjectList
from configs.dataset_categories import get_threshold, get_small_objects

from .meta_func.seg_func import SegFunc
from .meta_func.clip_func import ClipFunc
from .meta_func.core_func import CoreFunc
from .meta_func.flask_func import FlaskFunc
from .meta_func.io_func import IOFunc
from .meta_func.nav_func import NavFunc
from itertools import product



class SceneGraph(SegFunc, ClipFunc, CoreFunc, FlaskFunc, IOFunc, NavFunc):
    def __init__(self, map_resolution, map_size_cm, map_size, camera_matrix, is_navigation=True, agent=None, server_port=None, dataset='hm3d') -> None:
        # Initialize parent classes
        FlaskFunc.__init__(self, server_port=server_port)

        self.map_resolution = map_resolution
        self.map_size_cm = map_size_cm
        self.map_size = map_size
        full_w, full_h = self.map_size, self.map_size
        self.full_w = full_w
        self.full_h = full_h
        self.visited = torch.zeros(full_w, full_h).float().cpu().numpy()
        self.camera_matrix = camera_matrix
        self.SAM_ENCODER_VERSION = "vit_h"
        self.sam_variant = 'groundedsam'
        self.device = 'cuda'
        self.classes = ['item']
        self.BG_CLASSES = ["wall", "floor", "ceiling"]
        self.rooms = ['unknown', 'bedroom', 'living room', 'bathroom', 'kitchen', 'dining room', 'office room', 'gym', 'lounge', 'laundry room']
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.edge_list = []
        self.init_room_nodes()
        self.is_navigation = is_navigation
        self.seg_xyxy = None
        self.seg_caption = None

        self.groundingdino_config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.groundingdino_checkpoint = 'data/models/groundingdino_swint_ogc.pth'
        self.sam_version = 'vit_h'
        self.sam_checkpoint = 'data/models/sam_vit_h_4b8939.pth'
        self.segment2d_results = []

        # Dataset-specific configuration
        self.dataset = dataset
        print(f"[SceneGraph Init] Using dataset: {dataset.upper()}")

        # Load dataset-specific thresholds (DEPRECATED - will be handled dynamically)
        # Keep for backward compatibility but log warning
        self.threshold_list = {'bathtub': 2, 'bed': 5, 'cabinet': 3, 'chair': 5, 'chest_of_drawers': 5, 'clothes': 4, 'counter': 4, 'cushion': 4, 'fireplace': 4, 'gym_equipment': 5, 'picture': 4, 'plant': 3, 'seating': 2, 'shower': 2, 'sink': 3, 'sofa': 4, 'staircase': 3, 'stool': 5, 'table': 5, 'toilet': 2, 'towel': 4, 'tv_monitor': 2, 'treadmill. fitness equipment.': 0}

        # Load dataset-specific small objects list (unified naming)
        self.small_objects = get_small_objects(dataset)
        print(f"[SceneGraph Init] Small objects for {dataset.upper()}: {self.small_objects}")

        self.node_space = 'bathtub. bed. cabinet. chair. drawers. clothes. counter. cushion. fireplace. gym. picture. plant. seating. shower. sink. sofa. staircase. stool. table. toilet. towel. tv. treadmill. fitness equipment.'
        self.prompt_edge_proposal = '''
Provide the most possible single spatial relationship for each of the following object pairs. Answer with only one relationship per pair, and separate each answer with a newline character. Do not response superfluous text.
Example 1:
Input:
Object pair(s):
(cabinet, chair)
Output:
next to

Example 2:
Input:
Object pair(s):
(table, lamp)
(bed, nightstand)
Output:
on
next to

Now input is:
Object pair(s):
        '''
        self.prompt_relation = '''Describe the spatial relationship between the {} and the {} in the image.

IMPORTANT: Answer with ONLY a short spatial relationship phrase (2-4 words maximum).

Valid examples:
- next to
- on top of
- inside of
- under
- hang on

Your answer (spatial relationship only):'''
        self.prompt_discriminate_relation = 'In the image, do {} and {} satisfy the relationship of {}? Only answer "yes" or "no".'
        self.mask_generator = self.get_sam_mask_generator(self.sam_variant, self.device)
        self.set_cfg()
        self.set_agent(agent)

        # Enable open-vocabulary multi-class probability distribution
        # Set to True to get single-frame caption probabilities from GroundingDINO
        # Set to False to use multi-frame caption accumulation (old method)
        self.use_caption_distribution = False

    def get_threshold_for_category(self, category):
        """
        Get detection threshold for a specific category (expects unified naming).

        Args:
            category: Category name in unified naming (e.g., 'plant', 'tv_monitor', 'sofa')

        Returns:
            Threshold value (int)
        """
        return get_threshold(category, self.dataset)

    def reset(self):
        full_w, full_h = self.map_size, self.map_size
        self.full_w = full_w
        self.full_h = full_h
        self.visited = torch.zeros(full_w, full_h).float().cpu().numpy()
        self.segment2d_results = []
        self.reason = ''
        self.objects = MapObjectList(device=self.device)
        self.objects_post = MapObjectList(device=self.device)
        self.nodes = []
        self.init_room_nodes()
        self.edge_list = []
    
    def update_scenegraph(self):
        self.segment2d()
        if len(self.segment2d_results) > 0:
            self.mapping3d()
            self.get_caption(debug=False)  # Disable debug for speed
            self.update_node()
            self.update_edge()
            self.update_group()
            
    def get_scenegraph_elements(self):
        objects_post = self.objects_post
        # 收集所有 object nodes
        object_nodes = list(self.get_nodes())

        # 收集 room nodes 和 group nodes
        room_nodes = list(getattr(self, 'room_nodes', []))
        group_nodes = []
        for rn in room_nodes:
            group_nodes.extend(getattr(rn, 'group_nodes', []))
        edges = self.edge_list
        
        return objects_post, object_nodes, room_nodes, group_nodes, edges
    
    def enumerate_scene_graphs(self):
        """
        Enumerate all possible scene graphs based on object captions.

        Args:
            scenegraph: The scene graph object containing nodes with captions.

        Returns:
            List of scene graphs, where each scene graph is a list of objects with a specific caption.
        """
        # Extract all nodes from the scene graph
        nodes = self.nodes  # Assuming `scenegraph.nodes` contains all object nodes

        # Collect all possible captions for each node
        all_captions = [node.object['captions'] for node in nodes]

        # Generate all combinations of captions
        all_combinations = list(product(*all_captions))

        # Create new scene graphs for each combination
        scene_graphs = []
        for combination in all_combinations:
            new_scene_graph = []
            for i, caption in enumerate(combination):
                new_node = nodes[i].copy()  # Assuming nodes have a `copy` method
                new_node.set_caption(caption)
                new_scene_graph.append(new_node)
            scene_graphs.append(new_scene_graph)

        return scene_graphs
