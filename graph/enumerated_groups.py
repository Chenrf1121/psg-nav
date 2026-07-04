"""
EnumeratedGroupNodes: Data structure for storing enumerated GroupNode combinations

This module provides a convenient data structure to store and access
enumerated GroupNode results, considering caption distribution uncertainties.
"""

import numpy as np


class EnumeratedGroupNodes:
    """
    Container for enumerated GroupNode results.

    Stores all possible GroupNode caption combinations with their probabilities,
    providing both hierarchical (by room) and flat (sorted by probability) access.

    Attributes:
        all_enumerations (list): All enumerations sorted by probability (descending)
        by_room (dict): Enumerations grouped by room caption
        by_group (dict): Enumerations grouped by (room_caption, group_idx)

    Example:
        >>> enum_groups = EnumeratedGroupNodes()
        >>> enum_groups.add_from_room(room_node, top_k_per_group=3)
        >>> top_5 = enum_groups.get_top_k(5)
        >>> for enum_data in top_5:
        ...     print(enum_data['caption_combo'], enum_data['probability'])
    """

    def __init__(self):
        self.all_enumerations = []  # Flat list, sorted by probability
        self.by_room = {}  # room_caption -> list of enumeration dicts
        self.by_group = {}  # (room_caption, group_idx) -> list of enumeration dicts

    def add_from_room(self, room_node, enumerate_func, top_k_per_group=None, min_prob_threshold=None):
        """
        Enumerate and add GroupNodes from a RoomNode.

        Args:
            room_node: The RoomNode containing GroupNodes to enumerate
            enumerate_func: Function to enumerate a single GroupNode
                           Should have signature: enumerate_func(group_node) -> List[GroupNode]
            top_k_per_group: Optional, keep only top-k enumerations per GroupNode

        Returns:
            int: Number of enumerations added
        """
        if len(room_node.group_nodes) == 0:
            return 0

        added_count = 0
        room_enums = []

        for group_idx, group_node in enumerate(room_node.group_nodes):
            # Enumerate this GroupNode
            enumerated = enumerate_func(group_node, min_prob_threshold)

            # Optionally keep only top-k
            if top_k_per_group is not None:
                enumerated = enumerated[:top_k_per_group]

            # Normalize probabilities within top-k subset
            # After selecting top-k, normalize so that these k enumerations sum to 1.0
            if len(enumerated) > 0:
                total_prob = sum(gn.probability for gn in enumerated)
                if total_prob > 0:
                    for gn in enumerated:
                        gn.probability = gn.probability / total_prob

            # Store each enumeration
            for enum_idx, enum_group in enumerate(enumerated):
                enum_data = {
                    'room_caption': room_node.caption,
                    'room_node': room_node,
                    'group_idx': group_idx,
                    'original_group': group_node,
                    'enumeration_idx': enum_idx,
                    'group_node': enum_group,
                    'probability': enum_group.probability,
                    'caption_combo': [n.caption for n in enum_group.nodes],
                    'center': enum_group.center,
                }

                self.all_enumerations.append(enum_data)
                room_enums.append(enum_data)
                added_count += 1

            # Index by group
            group_key = (room_node.caption, group_idx)
            if group_key not in self.by_group:
                self.by_group[group_key] = []
            self.by_group[group_key].extend(enumerated[-len(enumerated):])  # Last batch

        # Index by room
        if room_node.caption not in self.by_room:
            self.by_room[room_node.caption] = []
        self.by_room[room_node.caption].extend(room_enums)

        # Sort after adding
        self._sort()

        return added_count

    def add_from_rooms(self, room_nodes, enumerate_func, top_k_per_group=None, min_prob_threshold=None):
        """
        Enumerate and add GroupNodes from multiple RoomNodes.

        Args:
            room_nodes: List of RoomNodes
            enumerate_func: Function to enumerate a single GroupNode
            top_k_per_group: Optional, keep only top-k enumerations per GroupNode

        Returns:
            int: Total number of enumerations added
        """
        total_added = 0
        for room_node in room_nodes:
            added = self.add_from_room(room_node, enumerate_func, top_k_per_group, min_prob_threshold)
            total_added += added
        return total_added

    def _sort(self):
        """Sort all_enumerations by probability (descending)."""
        self.all_enumerations.sort(key=lambda x: x['probability'], reverse=True)

    def get_top_k(self, k=10):
        """
        Get top-k enumerations by probability.

        Args:
            k: Number of top enumerations to return

        Returns:
            list: Top-k enumeration dicts
        """
        return self.all_enumerations[:k]

    def get_by_room(self, room_caption):
        """
        Get all enumerations for a specific room.

        Args:
            room_caption: Caption of the room (e.g., 'bedroom')

        Returns:
            list: List of enumeration dicts for this room
        """
        return self.by_room.get(room_caption, [])

    def get_by_group(self, room_caption, group_idx):
        """
        Get all enumerations for a specific GroupNode.

        Args:
            room_caption: Caption of the room
            group_idx: Index of the GroupNode within the room

        Returns:
            list: List of enumeration dicts for this GroupNode
        """
        key = (room_caption, group_idx)
        return self.by_group.get(key, [])

    def filter_by_probability(self, threshold=0.1):
        """
        Filter enumerations by minimum probability.

        Args:
            threshold: Minimum probability threshold

        Returns:
            list: Filtered enumeration dicts
        """
        return [e for e in self.all_enumerations if e['probability'] >= threshold]

    def get_room_captions(self):
        """Get list of all room captions with enumerations."""
        return list(self.by_room.keys())

    def get_stats(self):
        """
        Get statistics about the enumerations.

        Returns:
            dict: Statistics including counts and probability range
        """
        if len(self.all_enumerations) == 0:
            return {
                'total_enumerations': 0,
                'num_rooms': 0,
                'num_groups': 0,
                'prob_range': (0.0, 0.0),
            }

        return {
            'total_enumerations': len(self.all_enumerations),
            'num_rooms': len(self.by_room),
            'num_groups': len(self.by_group),
            'prob_range': (
                self.all_enumerations[-1]['probability'],  # Min
                self.all_enumerations[0]['probability']     # Max
            ),
        }

    def summary(self, verbose=False):
        """
        Print summary of enumeration results.

        Args:
            verbose: If True, print detailed information
        """
        stats = self.get_stats()

        print("\n" + "="*60)
        print("📊 Enumerated GroupNodes Summary")
        print("="*60)
        print(f"Total enumerations: {stats['total_enumerations']}")
        print(f"Rooms with GroupNodes: {stats['num_rooms']}")
        print(f"Total GroupNodes: {stats['num_groups']}")

        if stats['total_enumerations'] > 0:
            prob_min, prob_max = stats['prob_range']
            print(f"Probability range: {prob_min:.4f} - {prob_max:.4f}")

            if verbose:
                print(f"\nBreakdown by room:")
                for room_caption in sorted(self.by_room.keys()):
                    room_enums = self.by_room[room_caption]
                    print(f"  {room_caption}: {len(room_enums)} enumerations")

                print(f"\nTop-5 most probable enumerations:")
                for i, enum_data in enumerate(self.get_top_k(5)):
                    print(f"  {i+1}. {enum_data['caption_combo']} "
                          f"(prob: {enum_data['probability']:.4f}, "
                          f"room: {enum_data['room_caption']})")

        print("="*60 + "\n")

    def __len__(self):
        """Return total number of enumerations."""
        return len(self.all_enumerations)

    def __iter__(self):
        """Iterate over all enumerations (sorted by probability)."""
        return iter(self.all_enumerations)

    def __getitem__(self, idx):
        """Get enumeration by index (in probability-sorted order)."""
        return self.all_enumerations[idx]

    def apply_llm_filter(self, llm_filter, min_confidence=0.5):
        """
        Apply LLM-based filtering to remove implausible enumerations.

        Args:
            llm_filter: LLMFilter instance
            min_confidence: Minimum confidence threshold (not used currently)

        Returns:
            self (for method chaining)
        """
        llm_filter.filter_enumerated_groups(self, min_confidence=min_confidence)
        return self

    def __repr__(self):
        stats = self.get_stats()
        return (f"EnumeratedGroupNodes("
                f"total={stats['total_enumerations']}, "
                f"rooms={stats['num_rooms']}, "
                f"groups={stats['num_groups']})")
