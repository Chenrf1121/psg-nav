"""
Configuration file for object detector in PSG-Nav

Choose between GLIP and FastSAM detectors with different modes
"""

# ============================================
# Visualization Configuration
# ============================================
# Whether to use GroundingDINO masks in visualization
# If False, only show navigation detector (GLIP/FastSAM) results
USE_DINO_VISUALIZATION = False

# ============================================
# Detector Selection
# ============================================
# Detector type: 'glip', 'fastsam_clip', or 'fastsam_text'
# - 'glip': GLIP detector (original, high accuracy, ~600ms, 7GB VRAM)
# - 'fastsam_clip': FastSAM + CLIP batch matching (fast, ~305ms, 2.6GB VRAM)
# - 'fastsam_text': FastSAM + text_prompt loop (21 iterations, ~4500ms)
DETECTOR_TYPE = 'glip'  # Change to switch detector mode

# ============================================
# Model Weights Paths (Modify these paths as needed)
# ============================================
# GLIP weights
GLIP_WEIGHT_PATH = 'GLIP/MODEL/glip_large_model.pth'

# FastSAM weights - MODIFY THIS PATH
FASTSAM_WEIGHT_PATH = '/data/FastSAM/FastSAM-x.pt'
# FastSAM code path
FASTSAM_CODE_PATH = '/home/RufengChen/PSG-Nav/FastSAM'

# ============================================
# GLIP Detector Configuration
# ============================================
GLIP_CONFIG = {
    'config_file': 'GLIP/configs/pretrain/glip_Swin_L.yaml',
    'weight_file': GLIP_WEIGHT_PATH,
    'device': 'cuda',
    'min_image_size': 800,
    'confidence_threshold': 0.61,
}

# ============================================
# FastSAM Detector Configuration
# ============================================
# FastSAM + CLIP mode configuration
FASTSAM_CLIP_CONFIG = {
    'model_path': FASTSAM_WEIGHT_PATH,
    'device': 'cuda',
    'clip_model_name': 'ViT-B/32',  # Options: 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'
    'imgsz': 1024,
    'conf': 0.4,  # FastSAM confidence threshold
    'iou': 0.9,  # IoU threshold for NMS
    'retina_masks': True,
    'top_k_per_category': 1,  # Return top-k detections per category
    'clip_threshold': 0.6,  # Minimum CLIP similarity score (0-1)
    'min_area': 100,  # Minimum mask area in pixels
}

# FastSAM + text_prompt mode configuration
FASTSAM_TEXT_CONFIG = {
    'model_path': FASTSAM_WEIGHT_PATH,
    'device': 'cuda',
    'imgsz': 1024,
    'conf': 0.4,  # FastSAM confidence threshold
    'iou': 0.9,  # IoU threshold for NMS
    'retina_masks': True,
    'min_area': 100,  # Minimum mask area in pixels
    'text_conf_threshold': 0.5,  # Confidence threshold for text prompt results
}

# Backward compatibility: FASTSAM_CONFIG points to CLIP mode
FASTSAM_CONFIG = FASTSAM_CLIP_CONFIG

# ============================================
# Object Categories (Dynamic - use dataset_categories module)
# ============================================
# DEPRECATED: Use configs.dataset_categories module instead
# These are kept for backward compatibility but should not be used in new code

# Import category functions
from configs.dataset_categories import (
    get_categories,
    get_detection_categories,
    ROOM_CATEGORIES,
    DOOR_CATEGORIES
)

# For backward compatibility - will be MP3D categories by default
# Use get_categories(dataset) for proper dataset-specific categories
CATEGORIES_21 = [
    'chair', 'table', 'picture', 'cabinet', 'cushion',
    'sofa', 'bed', 'chest_of_drawers', 'plant', 'sink',
    'toilet', 'stool', 'towel', 'tv_monitor', 'shower',
    'bathtub', 'counter', 'fireplace', 'gym_equipment',
    'seating', 'clothes'
]
