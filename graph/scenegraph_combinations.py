"""
SceneGraphCombinations: Data structure for storing scene-graph level combinations

This module provides a data structure to store and manage combinations
of multiple RoomNodes across the entire scene graph, considering their joint probabilities.

Uses a heap-based Best-First Search algorithm to efficiently find top-k combinations
without generating all possible combinations (avoiding memory explosion).
"""

import heapq
import numpy as np


class SceneGraphCombinations:
    """
    Container for scene-graph level room combination results.

    Stores all possible combinations of room-level GroupNode combinations across
    the entire scene graph, where each combination represents one possible interpretation
    of the entire scene.

    Attributes:
        all_combinations (list): All combinations sorted by probability (descending)
        num_rooms (int): Number of rooms in each combination

    Example:
        >>> sg_combos = SceneGraphCombinations()
        >>> sg_combos.add_from_room_combinations(room_combinations, max_combinations=64)
        >>> top_10 = sg_combos.get_top_k(10)
    """

    def __init__(self):
        self.all_combinations = []  # Flat list, sorted by probability
        self.num_rooms = 0

    def add_from_room_combinations(self, room_combinations, max_combinations=64, min_prob_threshold=0.05):
        """
        Enumerate scene-graph level combinations from room-level combinations.

        Uses heap-based Best-First Search to efficiently find top-k combinations
        without generating all possible combinations.

        Args:
            room_combinations: RoomGroupCombinations instance with enumerated room combos
            max_combinations: Maximum number of scene-graph combinations to keep (default: 64)
            min_prob_threshold: Minimum joint probability threshold (default: 0.05)

        Returns:
            int: Number of combinations added
        """
        # Step 1: Collect room combinations grouped by room
        room_captions = room_combinations.get_room_captions()

        if not room_captions:
            return 0

        self.num_rooms = len(room_captions)

        # Collect and sort combinations for each room by probability
        room_combo_lists = []
        for room_caption in sorted(room_captions):  # Sort for consistency
            room_combos = room_combinations.get_by_room(room_caption)
            if room_combos:
                # Sort by probability descending and add log_prob
                sorted_combos = sorted(room_combos, key=lambda x: x['probability'], reverse=True)
                for c in sorted_combos:
                    prob = max(c['probability'], 1e-10)
                    c['log_prob'] = np.log(prob)
                room_combo_lists.append(sorted_combos)

        if not room_combo_lists:
            return 0

        num_rooms = len(room_combo_lists)

        # Step 2: Use heap-based Best-First Search
        # Initial state: best combination (index 0 for each room)
        initial_indices = tuple([0] * num_rooms)
        initial_log_prob_sum = sum(
            room_combo_lists[r][0]['log_prob'] for r in range(num_rooms)
        )

        # Max-heap simulation: store negative log_prob_sum
        heap = [(-initial_log_prob_sum, initial_indices)]
        visited = {initial_indices}
        visited_list = [initial_indices]  # Track insertion order for size limiting

        # Memory optimization: limit visited set size to prevent memory explosion
        MAX_VISITED_SIZE = 10000

        sg_combinations = []
        combo_idx = 0
        iteration_count = 0

        while heap and len(sg_combinations) < max_combinations:
            iteration_count += 1
            # Pop best combination
            neg_log_prob_sum, indices = heapq.heappop(heap)
            log_prob_sum = -neg_log_prob_sum
            joint_prob = np.exp(log_prob_sum)

            # Add if above threshold
            if joint_prob >= min_prob_threshold:
                room_combos_list = []
                for r, idx in enumerate(indices):
                    room_combos_list.append(room_combo_lists[r][idx])

                sg_combo_data = {
                    'room_combinations': room_combos_list,
                    'probability': joint_prob,
                    'num_rooms': len(room_combos_list),
                    'combination_idx': combo_idx,
                }
                sg_combinations.append(sg_combo_data)
                combo_idx += 1

            # Generate children
            for p in range(num_rooms):
                new_indices = list(indices)
                new_indices[p] += 1

                if new_indices[p] >= len(room_combo_lists[p]):
                    continue

                # Reset positions after p
                for q in range(p + 1, num_rooms):
                    new_indices[q] = 0

                new_indices = tuple(new_indices)

                if new_indices not in visited:
                    # Memory optimization: limit visited size
                    if len(visited) >= MAX_VISITED_SIZE:
                        oldest = visited_list.pop(0)
                        visited.discard(oldest)

                    visited.add(new_indices)
                    visited_list.append(new_indices)
                    new_log_prob_sum = sum(
                        room_combo_lists[r][new_indices[r]]['log_prob']
                        for r in range(num_rooms)
                    )
                    heapq.heappush(heap, (-new_log_prob_sum, new_indices))

        # Step 3: Expansion if too few pass threshold
        min_required = max(1, max_combinations // 4)
        if len(sg_combinations) < min_required:
            while heap and len(sg_combinations) < min_required:
                neg_log_prob_sum, indices = heapq.heappop(heap)
                log_prob_sum = -neg_log_prob_sum
                joint_prob = np.exp(log_prob_sum)

                room_combos_list = []
                for r, idx in enumerate(indices):
                    room_combos_list.append(room_combo_lists[r][idx])

                sg_combo_data = {
                    'room_combinations': room_combos_list,
                    'probability': joint_prob,
                    'num_rooms': len(room_combos_list),
                    'combination_idx': combo_idx,
                }
                sg_combinations.append(sg_combo_data)
                combo_idx += 1

                # Generate children for expansion
                for p in range(num_rooms):
                    new_indices = list(indices)
                    new_indices[p] += 1
                    if new_indices[p] >= len(room_combo_lists[p]):
                        continue
                    for q in range(p + 1, num_rooms):
                        new_indices[q] = 0
                    new_indices = tuple(new_indices)
                    if new_indices not in visited:
                        # Memory optimization: limit visited size
                        if len(visited) >= MAX_VISITED_SIZE:
                            oldest = visited_list.pop(0)
                            visited.discard(oldest)

                        visited.add(new_indices)
                        visited_list.append(new_indices)
                        new_log_prob_sum = sum(
                            room_combo_lists[r][new_indices[r]]['log_prob']
                            for r in range(num_rooms)
                        )
                        heapq.heappush(heap, (-new_log_prob_sum, new_indices))

        # Step 4: Normalize probabilities
        if sg_combinations:
            total_prob = sum(c['probability'] for c in sg_combinations)
            if total_prob > 0:
                for c in sg_combinations:
                    c['probability'] = c['probability'] / total_prob

        self.all_combinations = sg_combinations

        return len(sg_combinations)

    def get_top_k(self, k=10):
        """
        Get top-k scene-graph combinations by joint probability.

        Args:
            k: Number of top combinations to return

        Returns:
            list: Top-k combination dicts
        """
        return self.all_combinations[:k]

    def get_stats(self):
        """
        Get statistics about the scene-graph combinations.

        Returns:
            dict: Statistics including counts and probability range
        """
        if len(self.all_combinations) == 0:
            return {
                'total_combinations': 0,
                'num_rooms': 0,
                'prob_range': (0.0, 0.0),
            }

        return {
            'total_combinations': len(self.all_combinations),
            'num_rooms': self.num_rooms,
            'prob_range': (
                self.all_combinations[-1]['probability'],  # Min
                self.all_combinations[0]['probability']     # Max
            ),
        }

    def summary(self, verbose=False):
        """
        Print summary of scene-graph combination results.

        Args:
            verbose: If True, print detailed information
        """
        stats = self.get_stats()

        print("\n" + "="*70)
        print("🌐 Scene Graph Combinations Summary")
        print("="*70)
        print(f"Total combinations: {stats['total_combinations']}")
        print(f"Number of rooms: {stats['num_rooms']}")

        if stats['total_combinations'] > 0:
            prob_min, prob_max = stats['prob_range']
            print(f"Probability range: {prob_min:.6f} - {prob_max:.6f}")

            if verbose:
                print(f"\nTop-5 most probable scene-graph combinations:")
                for i, sg_combo in enumerate(self.get_top_k(5)):
                    print(f"\n  {i+1}. Probability: {sg_combo['probability']:.6f}")
                    print(f"     Rooms: {sg_combo['num_rooms']}")

                    # Print room-level details
                    for room_combo in sg_combo['room_combinations']:
                        room_caption = room_combo['room_caption']
                        num_groups = room_combo['num_groups']
                        print(f"       - {room_caption}: {num_groups} groups")

        print("="*70 + "\n")

    def __len__(self):
        """Return total number of scene-graph combinations."""
        return len(self.all_combinations)

    def __iter__(self):
        """Iterate over all combinations (sorted by probability)."""
        return iter(self.all_combinations)

    def __getitem__(self, idx):
        """Get combination by index (in probability-sorted order)."""
        return self.all_combinations[idx]

    def __repr__(self):
        stats = self.get_stats()
        return (f"SceneGraphCombinations("
                f"total={stats['total_combinations']}, "
                f"rooms={stats['num_rooms']})")
