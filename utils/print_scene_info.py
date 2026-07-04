"""
Helper script to print scene and episode information during navigation.
Add this to your agent's act() method to see the actual scene/episode IDs.
"""

def print_scene_info(agent):
    """
    Print current scene and episode information.

    Usage: Add this line in your agent's act() method:
        from utils.print_scene_info import print_scene_info
        print_scene_info(self)
    """
    try:
        current_episode = agent.simulator._env.current_episode
        scene_id = current_episode.scene_id if hasattr(current_episode, 'scene_id') else 'N/A'
        episode_id = current_episode.episode_id if hasattr(current_episode, 'episode_id') else 'N/A'

        # Extract clean scene name from path
        scene_name = scene_id.split('/')[-1].split('.')[0] if scene_id != 'N/A' else 'N/A'

        print(f"\n{'='*60}")
        print(f"Scene Info - Step {agent.navigate_steps}")
        print(f"{'='*60}")
        print(f"Scene ID (full): {scene_id}")
        print(f"Scene ID (clean): {scene_name}")
        print(f"Episode ID: {episode_id}")
        print(f"Object Goal: {agent.obj_goal}")
        print(f"Current Disk Radius: {agent.map_manager.get_disk_radius()}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error getting scene info: {e}")
