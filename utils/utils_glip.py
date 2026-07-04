import csv
import gzip
import json
import copy
from configs.dataset_categories import (
    get_categories, get_detection_categories,
    ROOM_CATEGORIES, DOOR_CATEGORIES, EXTRA_OBJECTS,
    uses_episode_goal_categories,
)

# ============================================
# Dataset-agnostic Global Variables
# ============================================
# These will be initialized by init_categories(dataset)

categories = []  # categories except doors (from Matterport mapping)
categories_40 = []
categories_map = {}
categories_doors = []

# Load Matterport category mappings (for room detection, etc.)
with open('tools/matterport_category_mappings.tsv') as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for i, line in enumerate(tsv_file):
        line_ = [item for item in line[0].split('   ') if not item =='']
        if i == 0 or len(line_) < 4:
            continue
        if int(line_[3]) > 10:
            if 'door' in line_[-1] and line_[2] not in categories_doors:
                categories_doors.append(line_[2])
            else:
                categories.append(line_[2])
                categories_map[line_[2]] = line_[-1]
        if line_[-1] not in categories_40 and line_[-1] is not 'objects' and 'void' not in line_[-1]:
            categories_40.append(line_[-1])

# ============================================
# Dataset-specific Categories (Dynamic)
# ============================================
# Default to HM3D, will be overridden by init_categories()

categories_21_origin = None  # Base categories for current dataset
categories_21 = None  # Base categories + extra objects
object_captions = None  # Caption string for GLIP detection
projection = {}  # Category ID to name mapping
projection_reverse = {}  # Category name to ID mapping
current_dataset = None

# Shared categories (dataset-agnostic)
rooms = ROOM_CATEGORIES
rooms_captions = '. '.join(rooms) + '.'
door_captions = '. '.join(DOOR_CATEGORIES) + '.'


def init_categories(dataset='hm3d', goal_category=None):
    """
    Initialize category lists and captions based on dataset.
    Should be called once at the beginning with the correct dataset name.

    Args:
        dataset: 'mp3d', 'hm3d', 'hm3d_v2', 'hssd', or 'hm3d_ovon'
        goal_category: Optional episode goal category for open-vocabulary datasets
    """
    global categories_21_origin, categories_21, object_captions
    global projection, projection_reverse, current_dataset

    print(f"[Category Init] Initializing categories for dataset: {dataset.upper()}")
    current_dataset = dataset.lower()

    # Get base categories (unified naming)
    categories_21_origin = get_categories(dataset, goal_category=goal_category)

    # Add extra detection objects
    categories_21 = categories_21_origin + EXTRA_OBJECTS

    # Create caption string for GLIP
    object_captions = '. '.join(categories_21) + '.'

    # Load projection mapping from dataset-specific file
    dataset_lower = dataset.lower()
    if dataset_lower == 'mp3d':
        mapping_file = "tools/val.json.gz"
    elif dataset_lower in ['hm3d', 'hm3d_v2']:
        mapping_file = "tools/val_hm3d.json.gz"
    elif dataset_lower == 'hssd':
        mapping_file = "tools/val_hssd.json.gz"
    elif dataset_lower == 'hm3d_ovon':
        mapping_file = None
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if mapping_file:
        try:
            with gzip.open(mapping_file, 'r') as fin:
                json_bytes = fin.read()
            json_str = json_bytes.decode('utf-8')
            data = json.loads(json_str)
            projection_reverse = data['category_to_task_category_id']
        except FileNotFoundError:
            print(f"[Category Init] Warning: {mapping_file} not found, using default mapping")
            projection_reverse = {cat: idx for idx, cat in enumerate(categories_21_origin)}
    else:
        projection_reverse = {cat: idx for idx, cat in enumerate(categories_21_origin)}

    # Create reverse mapping
    projection = {}
    for key, item in projection_reverse.items():
        projection[item] = key

    print(f"[Category Init] Loaded {len(categories_21_origin)} base categories: {categories_21_origin}")
    print(f"[Category Init] Total detection categories (with extras): {len(categories_21)}")


def set_episode_goal_category(goal_category, dataset=None):
    """Refresh runtime detection categories for episode-specific open-vocabulary goals."""
    dataset = (dataset or current_dataset or 'hm3d').lower()
    if not uses_episode_goal_categories(dataset):
        return
    init_categories(dataset, goal_category=goal_category)


# Initialize with default dataset (HM3D)
# This will be overridden by calling init_categories() with the correct dataset
init_categories('hm3d')

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = {'x1': bb1[0], 'x2': bb1[2], 'y1': bb1[1], 'y2': bb1[3]}
    bb2 = {'x1': bb2[0], 'x2': bb2[2], 'y1': bb2[1], 'y2': bb2[3]}
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
