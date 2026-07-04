"""
RoomGroupCombinations: Data structure for storing combinations of GroupNodes within rooms

This module provides a data structure to store and manage combinations
of multiple GroupNodes within the same room, considering their joint probabilities.

Uses a heap-based Best-First Search algorithm to efficiently find top-k combinations
without generating all possible combinations (avoiding memory explosion).
"""

import heapq
import numpy as np

class RoomGroupCombinations:
    """
    Container for room-level GroupNode combination results.

    Stores all possible combinations of GroupNodes within each room, where each
    combination represents one possible interpretation of all GroupNodes in that room.

    For example, if a bedroom has 2 GroupNodes:
        - GroupNode1 can be: [bed, table] or [bed, chair]
        - GroupNode2 can be: [lamp] or [clock]

    This will generate 4 combinations:
        1. ([bed, table], [lamp])
        2. ([bed, table], [clock])
        3. ([bed, chair], [lamp])
        4. ([bed, chair], [clock])

    Attributes:
        all_combinations (list): All combinations sorted by probability (descending)
        by_room (dict): Combinations grouped by room caption

    Example:
        >>> combinations = RoomGroupCombinations()
        >>> combinations.add_from_room(room_node, scenegraph.enumerate_group_node, top_k_per_group=3)
        >>> top_10 = combinations.get_top_k(10)
        >>> for combo in top_10:
        ...     print(f"Room: {combo['room_caption']}, Prob: {combo['probability']:.4f}")
    """

    def __init__(self):
        self.all_combinations = []  # Flat list, sorted by probability
        self.by_room = {}  # room_caption -> list of combination dicts

    def add_from_room(self, room_node, enumerate_func, top_k_per_group=None, top_k=32):
        """
        Enumerate and add GroupNode combinations from a single RoomNode.

        Uses heap-based Best-First Search to efficiently find top-k combinations.

        Args:
            room_node: The RoomNode containing GroupNodes to enumerate
            enumerate_func: Function to enumerate a single GroupNode
                           Should have signature: enumerate_func(group_node) -> List[GroupNode]
            top_k_per_group: Optional, keep only top-k enumerations per GroupNode
            top_k: Maximum number of combinations to keep (default: 32)

        Returns:
            int: Number of combinations added
        """
        if len(room_node.group_nodes) == 0:
            return 0

        # Enumerate each GroupNode independently and sort by probability
        all_enumerations = []
        for group_node in room_node.group_nodes:
            enumerated = enumerate_func(group_node)

            # Optionally keep only top-k per group
            if top_k_per_group is not None:
                enumerated = enumerated[:top_k_per_group]

            # Sort by probability descending and add log_prob
            sorted_enums = sorted(enumerated, key=lambda x: x.probability, reverse=True)
            for e in sorted_enums:
                prob = max(e.probability, 1e-10)
                e.log_prob = np.log(prob)
            all_enumerations.append(sorted_enums)

        if not all_enumerations:
            return 0

        num_groups = len(all_enumerations)

        # Use heap-based Best-First Search
        initial_indices = tuple([0] * num_groups)
        initial_log_prob_sum = sum(
            all_enumerations[g][0].log_prob for g in range(num_groups)
        )

        heap = [(-initial_log_prob_sum, initial_indices)]
        visited = {initial_indices}

        room_combos = []
        combo_idx = 0

        while heap and len(room_combos) < top_k:
            neg_log_prob_sum, indices = heapq.heappop(heap)
            log_prob_sum = -neg_log_prob_sum
            combined_prob = np.exp(log_prob_sum)

            # Build combination
            group_combo = [all_enumerations[g][indices[g]] for g in range(num_groups)]

            combo_data = {
                'room_caption': room_node.caption,
                'room_node': room_node,
                'group_nodes': group_combo,
                'probability': combined_prob,
                'num_groups': len(group_combo),
                'combination_idx': combo_idx,
            }

            self.all_combinations.append(combo_data)
            room_combos.append(combo_data)
            combo_idx += 1

            # Generate children
            for p in range(num_groups):
                new_indices = list(indices)
                new_indices[p] += 1

                if new_indices[p] >= len(all_enumerations[p]):
                    continue

                for q in range(p + 1, num_groups):
                    new_indices[q] = 0

                new_indices = tuple(new_indices)

                if new_indices not in visited:
                    visited.add(new_indices)
                    new_log_prob_sum = sum(
                        all_enumerations[g][new_indices[g]].log_prob
                        for g in range(num_groups)
                    )
                    heapq.heappush(heap, (-new_log_prob_sum, new_indices))

        # Index by room
        if room_node.caption not in self.by_room:
            self.by_room[room_node.caption] = []
        self.by_room[room_node.caption].extend(room_combos)

        # Sort after adding
        self._sort()

        return len(room_combos)

    def add_from_rooms(self, room_nodes, enumerate_func, top_k_per_group=None):
        """
        Enumerate and add GroupNode combinations from multiple RoomNodes.

        Args:
            room_nodes: List of RoomNodes
            enumerate_func: Function to enumerate a single GroupNode
            top_k_per_group: Optional, keep only top-k enumerations per GroupNode

        Returns:
            int: Total number of combinations added
        """
        total_added = 0
        for room_node in room_nodes:
            added = self.add_from_room(room_node, enumerate_func, top_k_per_group)
            total_added += added
        return total_added


    def add_from_filtered_enumerations(self, enumerated_groups, top_k=None, min_prob_threshold=0.0):
        """
        Build room combinations from already filtered GroupNode enumerations.

        This ensures that only the plausible GroupNode enumerations (after LLM filtering)
        are used to construct room-level combinations.

        Args:
            enumerated_groups: EnumeratedGroupNodes instance (already filtered)
            top_k: Optional, keep only top-k combinations per room (default: 32)
            min_prob_threshold: Minimum probability threshold for filtering (default: 0.1)

        Returns:
            int: Total number of combinations added
        """
        total_added = 0
        top_k = top_k if top_k is not None else 16

        # Get all unique rooms from enumerated groups
        room_captions = set()
        for enum_data in enumerated_groups.all_enumerations:
            room_captions.add(enum_data['room_caption'])

        # Process each room independently
        for room_caption in room_captions:
            # Enumerate combinations for this single room
            room_combos = self._enumerate_single_room(
                room_caption=room_caption,
                enumerated_groups=enumerated_groups,
                top_k=top_k,
                min_prob_threshold=min_prob_threshold
            )

            # Add to container
            if room_combos:
                for combo_data in room_combos:
                    self.all_combinations.append(combo_data)
                    total_added += 1

                # Index by room
                if room_caption not in self.by_room:
                    self.by_room[room_caption] = []
                self.by_room[room_caption].extend(room_combos)

        # Sort all combinations by probability
        self._sort()

        return total_added

    def _enumerate_single_room(self, room_caption, enumerated_groups, top_k=32, min_prob_threshold=0.1):
        """
        Enumerate top-k GroupNode combinations for a single room using heap-based Best-First Search.

        This algorithm avoids generating all combinations by using a priority queue to explore
        only the most promising combinations. Uses log-probability to avoid float underflow.

        Args:
            room_caption: Caption of the room to enumerate
            enumerated_groups: EnumeratedGroupNodes instance
            top_k: Maximum number of combinations to keep
            min_prob_threshold: Minimum joint probability threshold

        Returns:
            list: List of combination dicts with normalized probabilities
        """
        # Step 1: Collect enumerated GroupNodes for this room
        groups_dict = {}
        room_node = None

        for enum_data in enumerated_groups.all_enumerations:
            if enum_data['room_caption'] == room_caption:
                group_idx = enum_data['group_idx']
                if group_idx not in groups_dict:
                    groups_dict[group_idx] = []
                groups_dict[group_idx].append(enum_data)

                if room_node is None:
                    room_node = enum_data['room_node']

        if not groups_dict or room_node is None:
            return []

        # Step 2: Sort each group's enumerations by probability (descending)
        # and convert to log-probability to avoid underflow
        sorted_group_indices = sorted(groups_dict.keys())
        group_enumerations = []

        for group_idx in sorted_group_indices:
            enums = groups_dict[group_idx]
            # Sort by probability descending
            sorted_enums = sorted(enums, key=lambda x: x['probability'], reverse=True)
            # Add log_prob to each enum
            for e in sorted_enums:
                prob = max(e['probability'], 1e-10)  # Avoid log(0)
                e['log_prob'] = np.log(prob)
            group_enumerations.append(sorted_enums)

        num_groups = len(group_enumerations)

        # Step 3: Use heap-based Best-First Search to find top-k combinations
        # State: (neg_log_prob_sum, indices_tuple)
        # We use negative log_prob because heapq is a min-heap

        # Initial state: best combination (index 0 for each group)
        initial_indices = tuple([0] * num_groups)
        initial_log_prob_sum = sum(
            group_enumerations[g][0]['log_prob'] for g in range(num_groups)
        )

        # Max-heap simulation: store negative log_prob_sum
        heap = [(-initial_log_prob_sum, initial_indices)]
        visited = {initial_indices}
        visited_list = [initial_indices]  # Track insertion order for size limiting

        # Memory optimization: limit visited set size to prevent memory explosion
        MAX_VISITED_SIZE = 10000

        room_combos = []
        combo_idx = 0
        iteration_count = 0

        while heap and len(room_combos) < top_k:
            iteration_count += 1
            # Pop best combination
            neg_log_prob_sum, indices = heapq.heappop(heap)
            log_prob_sum = -neg_log_prob_sum

            # Convert log_prob back to probability
            joint_prob = np.exp(log_prob_sum)

            # Skip if below threshold (but we still need to explore children)
            if joint_prob >= min_prob_threshold:
                # Build combination data
                group_nodes = []
                for g, idx in enumerate(indices):
                    group_nodes.append(group_enumerations[g][idx]['group_node'])

                combo_data = {
                    'room_caption': room_caption,
                    'room_node': room_node,
                    'group_nodes': group_nodes,
                    'probability': joint_prob,
                    'num_groups': len(group_nodes),
                    'combination_idx': combo_idx,
                }
                room_combos.append(combo_data)
                combo_idx += 1

            # Generate children: increment each index position
            # Child generation: for position p, increment index[p] and reset all positions after p
            for p in range(num_groups):
                new_indices = list(indices)
                new_indices[p] += 1

                # Check bounds
                if new_indices[p] >= len(group_enumerations[p]):
                    continue

                # For positions after p, reset to the minimum valid index
                # This ensures we explore in the correct order
                for q in range(p + 1, num_groups):
                    new_indices[q] = 0

                new_indices = tuple(new_indices)

                if new_indices not in visited:
                    # Memory optimization: limit visited size
                    if len(visited) >= MAX_VISITED_SIZE:
                        oldest = visited_list.pop(0)
                        visited.discard(oldest)

                    visited.add(new_indices)
                    visited_list.append(new_indices)
                    # Calculate new log_prob_sum
                    new_log_prob_sum = sum(
                        group_enumerations[g][new_indices[g]]['log_prob']
                        for g in range(num_groups)
                    )
                    heapq.heappush(heap, (-new_log_prob_sum, new_indices))

        # Step 4: Expansion strategy - if too few combos pass threshold
        min_required = max(1, top_k // 4)
        if len(room_combos) < min_required:
            # Continue extracting from heap even if below threshold
            while heap and len(room_combos) < min_required:
                neg_log_prob_sum, indices = heapq.heappop(heap)
                log_prob_sum = -neg_log_prob_sum
                joint_prob = np.exp(log_prob_sum)

                group_nodes = []
                for g, idx in enumerate(indices):
                    group_nodes.append(group_enumerations[g][idx]['group_node'])

                combo_data = {
                    'room_caption': room_caption,
                    'room_node': room_node,
                    'group_nodes': group_nodes,
                    'probability': joint_prob,
                    'num_groups': len(group_nodes),
                    'combination_idx': combo_idx,
                }
                room_combos.append(combo_data)
                combo_idx += 1

                # Generate children for expansion
                for p in range(num_groups):
                    new_indices = list(indices)
                    new_indices[p] += 1
                    if new_indices[p] >= len(group_enumerations[p]):
                        continue
                    for q in range(p + 1, num_groups):
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
                            group_enumerations[g][new_indices[g]]['log_prob']
                            for g in range(num_groups)
                        )
                        heapq.heappush(heap, (-new_log_prob_sum, new_indices))

        # Step 5: Normalize probabilities
        if room_combos:
            total_prob = sum(c['probability'] for c in room_combos)
            if total_prob > 0:
                for c in room_combos:
                    c['probability'] = c['probability'] / total_prob

        return room_combos

    def _sort(self):
        """Sort all_combinations by probability (descending)."""
        self.all_combinations.sort(key=lambda x: x['probability'], reverse=True)

    def get_top_k(self, k=10):
        """
        Get top-k combinations by joint probability.

        Args:
            k: Number of top combinations to return

        Returns:
            list: Top-k combination dicts
        """
        return self.all_combinations[:k]

    def get_by_room(self, room_caption):
        """
        Get all combinations for a specific room.

        Args:
            room_caption: Caption of the room (e.g., 'bedroom')

        Returns:
            list: List of combination dicts for this room, sorted by probability
        """
        room_combos = self.by_room.get(room_caption, [])
        # Sort by probability
        return sorted(room_combos, key=lambda x: x['probability'], reverse=True)

    def filter_by_probability(self, threshold=0.1):
        """
        Filter combinations by minimum joint probability.

        Args:
            threshold: Minimum probability threshold

        Returns:
            list: Filtered combination dicts
        """
        return [c for c in self.all_combinations if c['probability'] >= threshold]

    def filter_by_num_groups(self, min_groups=None, max_groups=None):
        """
        Filter combinations by number of GroupNodes.

        Args:
            min_groups: Minimum number of GroupNodes in combination
            max_groups: Maximum number of GroupNodes in combination

        Returns:
            list: Filtered combination dicts
        """
        filtered = self.all_combinations

        if min_groups is not None:
            filtered = [c for c in filtered if c['num_groups'] >= min_groups]

        if max_groups is not None:
            filtered = [c for c in filtered if c['num_groups'] <= max_groups]

        return filtered

    def get_room_captions(self):
        """Get list of all room captions with combinations."""
        return list(self.by_room.keys())

    def get_stats(self):
        """
        Get statistics about the combinations.

        Returns:
            dict: Statistics including counts and probability range
        """
        if len(self.all_combinations) == 0:
            return {
                'total_combinations': 0,
                'num_rooms': 0,
                'avg_groups_per_combo': 0.0,
                'prob_range': (0.0, 0.0),
            }

        total_groups = sum(c['num_groups'] for c in self.all_combinations)

        return {
            'total_combinations': len(self.all_combinations),
            'num_rooms': len(self.by_room),
            'avg_groups_per_combo': total_groups / len(self.all_combinations),
            'prob_range': (
                self.all_combinations[-1]['probability'],  # Min
                self.all_combinations[0]['probability']     # Max
            ),
        }

    def summary(self, verbose=False):
        """
        Print summary of combination results.

        Args:
            verbose: If True, print detailed information
        """
        stats = self.get_stats()

        print("\n" + "="*70)
        print("🏠 Room GroupNode Combinations Summary")
        print("="*70)
        print(f"Total combinations: {stats['total_combinations']}")
        print(f"Rooms with combinations: {stats['num_rooms']}")
        print(f"Average GroupNodes per combination: {stats['avg_groups_per_combo']:.2f}")

        if stats['total_combinations'] > 0:
            prob_min, prob_max = stats['prob_range']
            print(f"Probability range: {prob_min:.6f} - {prob_max:.6f}")

            if verbose:
                print(f"\nBreakdown by room:")
                for room_caption in sorted(self.by_room.keys()):
                    room_combos = self.by_room[room_caption]
                    print(f"  {room_caption}: {len(room_combos)} combinations")

                print(f"\nTop-5 most probable combinations:")
                for i, combo_data in enumerate(self.get_top_k(5)):
                    group_info = []
                    for group in combo_data['group_nodes']:
                        captions = [n.caption for n in group.nodes]
                        group_info.append(f"[{', '.join(captions)}]")

                    print(f"  {i+1}. Room: {combo_data['room_caption']}")
                    print(f"     Groups: {' + '.join(group_info)}")
                    print(f"     Probability: {combo_data['probability']:.6f}")

        print("="*70 + "\n")

    def get_combination_details(self, combination_data):
        """
        Get human-readable details for a combination.

        Args:
            combination_data: A combination dict from this container

        Returns:
            str: Formatted string describing the combination
        """
        lines = []
        lines.append(f"Room: {combination_data['room_caption']}")
        lines.append(f"Probability: {combination_data['probability']:.6f}")
        lines.append(f"Number of GroupNodes: {combination_data['num_groups']}")
        lines.append("GroupNodes:")

        for idx, group in enumerate(combination_data['group_nodes']):
            captions = [n.caption for n in group.nodes]
            lines.append(f"  Group {idx+1}: [{', '.join(captions)}] (prob: {group.probability:.4f})")

        return '\n'.join(lines)

    def __len__(self):
        """Return total number of combinations."""
        return len(self.all_combinations)

    def __iter__(self):
        """Iterate over all combinations (sorted by probability)."""
        return iter(self.all_combinations)

    def __getitem__(self, idx):
        """Get combination by index (in probability-sorted order)."""
        return self.all_combinations[idx]

    def apply_llm_filter(self, llm_filter, min_confidence=0.5):
        """
        Apply LLM-based filtering to remove implausible room combinations.

        Args:
            llm_filter: LLMFilter instance
            min_confidence: Minimum confidence threshold (not used currently)

        Returns:
            self (for method chaining)
        """
        llm_filter.filter_room_combinations(self, min_confidence=min_confidence)
        return self

    def __repr__(self):
        stats = self.get_stats()
        return (f"RoomGroupCombinations("
                f"total={stats['total_combinations']}, "
                f"rooms={stats['num_rooms']}, "
                f"avg_groups={stats['avg_groups_per_combo']:.1f})")
