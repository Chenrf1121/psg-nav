"""
Helper script to collect scene_id mappings from your navigation runs.
Run this to automatically generate the scene_name_mapping dictionary.

Usage:
    1. Add this at the beginning of your agent's reset() or act() method:
       from utils.collect_scene_mappings import log_scene_info
       log_scene_info(self)

    2. Run your navigation experiments

    3. Run this script to generate the mapping:
       python utils/collect_scene_mappings.py

    4. Copy the output to map_manager.py's scene_name_mapping
"""

import os
from collections import OrderedDict


def log_scene_info(agent):
    """
    Log scene information to a file. Call this in your agent's reset() method.

    Args:
        agent: The PSG_Nav_Agent instance
    """
    try:
        current_episode = agent.simulator._env.current_episode
        if hasattr(current_episode, 'scene_id'):
            scene_id = current_episode.scene_id.split('/')[-1].split('.')[0]

            # Append to log file
            with open('scene_mappings.log', 'a') as f:
                f.write(f"{scene_id}\n")

    except Exception as e:
        pass


def generate_mapping():
    """
    Generate scene_name_mapping from collected logs.
    """
    log_file = 'scene_mappings.log'

    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        print("Please add log_scene_info() to your agent's reset() method first.")
        return

    # Read unique scene IDs
    scene_ids = []
    with open(log_file, 'r') as f:
        scene_ids = list(OrderedDict.fromkeys([line.strip() for line in f if line.strip()]))

    if not scene_ids:
        print("No scene IDs found in log file.")
        return

    print("\n" + "="*70)
    print("Scene Name Mapping - Copy this to map_manager.py")
    print("="*70 + "\n")

    print("self.scene_name_mapping = {")
    for i, scene_id in enumerate(scene_ids, 1):
        print(f"    'scene_{i:03d}': '{scene_id}',")
    print("}")

    print("\n" + "="*70)
    print(f"\nTotal scenes found: {len(scene_ids)}")
    print(f"Scene IDs: {', '.join(scene_ids)}")
    print("="*70 + "\n")

    # Also generate a reference table
    print("\nReference Table:")
    print("-" * 70)
    print(f"{'Friendly Name':<20} {'Actual Scene ID':<30}")
    print("-" * 70)
    for i, scene_id in enumerate(scene_ids, 1):
        print(f"scene_{i:03d:<16} {scene_id}")
    print("-" * 70)


if __name__ == "__main__":
    generate_mapping()
