"""
Dataset-specific category configurations for ObjectNav benchmarks.

This module defines object categories, thresholds, and mappings for HSSD ObjectNav.
"""

from configs.dataset_registry import is_open_vocabulary_dataset

HSSD_CATEGORIES = [
    'chair', 'bed', 'potted_plant', 'toilet', 'tv', 'couch'
]

HSSD_TO_UNIFIED = {
    'potted_plant': 'plant',
    'tv': 'tv_monitor',
    'couch': 'sofa',
}

UNIFIED_TO_HSSD = {
    'plant': 'potted_plant',
    'tv_monitor': 'tv',
    'sofa': 'couch',
}

HSSD_SMALL_OBJECTS = ['potted_plant', 'tv']

HSSD_THRESHOLDS = {
    'chair': 5, 'bed': 5, 'potted_plant': 3, 'toilet': 2, 'tv': 2, 'couch': 4
}

EXTRA_OBJECTS = ['heater', 'window', 'treadmill', 'exercise machine', 'staircase']
ROOM_CATEGORIES = [
    'bedroom', 'living room', 'bathroom', 'kitchen', 'dining room',
    'office room', 'gym', 'lounge', 'laundry room'
]
DOOR_CATEGORIES = ['doorway', 'hallway']

def get_categories(dataset='hssd', goal_category=None):
    """
    Get HSSD ObjectNav categories in unified internal naming.
    """
    dataset = dataset.lower()
    if dataset == 'hssd':
        return [HSSD_TO_UNIFIED.get(cat, cat) for cat in HSSD_CATEGORIES]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_detection_categories(dataset='hssd', goal_category=None):
    """
    Get categories for GLIP/FastSAM detection prompts.
    """
    base_categories = get_categories(dataset, goal_category=goal_category)
    return base_categories + EXTRA_OBJECTS


def get_small_objects(dataset='hssd'):
    """
    Get HSSD small object categories in unified internal naming.
    """
    dataset = dataset.lower()
    if dataset == 'hssd':
        return [HSSD_TO_UNIFIED.get(cat, cat) for cat in HSSD_SMALL_OBJECTS]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_threshold(category, dataset='hssd'):
    """
    Get HSSD detection threshold for a category.
    """
    dataset = dataset.lower()

    if dataset == 'hssd':
        hssd_cat = UNIFIED_TO_HSSD.get(category, category)
        return HSSD_THRESHOLDS.get(hssd_cat, 3)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def unified_to_dataset_name(category, dataset='hssd'):
    """
    Convert unified category name to HSSD category name.
    """
    dataset = dataset.lower()
    if dataset == 'hssd':
        return UNIFIED_TO_HSSD.get(category, category)
    raise ValueError(f"Unknown dataset: {dataset}")


def dataset_to_unified_name(category, dataset='hssd'):
    """
    Convert HSSD category name to unified category name.
    """
    dataset = dataset.lower()
    if dataset == 'hssd':
        return HSSD_TO_UNIFIED.get(category, category)
    raise ValueError(f"Unknown dataset: {dataset}")


def get_category_count(dataset='hssd', goal_category=None):
    """Get the number of HSSD ObjectNav categories."""
    return len(get_categories(dataset, goal_category=goal_category))


def uses_episode_goal_categories(dataset='hssd'):
    """Whether this dataset should derive detection categories from each episode goal."""
    return is_open_vocabulary_dataset(dataset)
