"""
Central registry for supported evaluation datasets.
"""

DATASET_CONFIGS = {
    "hssd": {
        "config_path": "configs/challenge_objectnav_hssd.local.rgbd.yaml",
        "default_split": "val",
        "fallback_scene_count": 40,
        "open_vocabulary": False,
    },
}

SUPPORTED_DATASETS = tuple(DATASET_CONFIGS.keys())


def get_dataset_info(dataset: str):
    dataset = dataset.lower()
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Supported datasets: {list(SUPPORTED_DATASETS)}"
        )
    return DATASET_CONFIGS[dataset]


def get_dataset_config_path(dataset: str) -> str:
    return get_dataset_info(dataset)["config_path"]


def get_default_dataset_split(dataset: str) -> str:
    return get_dataset_info(dataset)["default_split"]


def get_fallback_scene_count(dataset: str):
    return get_dataset_info(dataset)["fallback_scene_count"]


def is_open_vocabulary_dataset(dataset: str) -> bool:
    return bool(get_dataset_info(dataset)["open_vocabulary"])
