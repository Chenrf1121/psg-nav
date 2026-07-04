"""
Simple LLM Filter for Scene Graph Enumerations

Uses local Flask API to filter implausible GroupNode enumerations and room combinations.
"""

import json
import re
from typing import List, Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.meta_func.flask_func import FlaskFunc


class LLMFilter(FlaskFunc):
    """
    Simple LLM filter using local Flask API for filtering scene graph enumerations.

    This class uses prompt templates to ask the LLM whether object combinations
    are plausible, then filters out implausible results.
    """

    def __init__(self, verbose=False):
        """
        Initialize LLM filter.

        Args:
            verbose: Whether to print debug information
        """
        super().__init__()
        self.verbose = verbose

        # Prompt templates
        self.group_filter_template = """You are evaluating whether object groups in indoor scenes are plausible.

Given a room type and a list of objects found together in a group, determine if this combination is reasonable.

Room: {room}
Objects in group: {objects}

Is this a plausible combination? Answer in JSON format:
{{"plausible": true/false, "reason": "brief explanation"}}

Answer:"""

        self.room_filter_template = """You are evaluating whether a room configuration is plausible.

Given a room type and multiple object groups within that room, determine if the overall configuration is reasonable.

Room: {room}
Object groups:
{groups}

Is this a plausible room configuration? Answer in JSON format:
{{"plausible": true/false, "reason": "brief explanation"}}

Answer:"""

        # Statistics
        self.stats = {
            'group_filtered': 0,
            'group_kept': 0,
            'room_filtered': 0,
            'room_kept': 0,
        }

    def filter_enumerated_groups(self, enumerated_groups, min_confidence=0.5):
        """
        Filter implausible GroupNode enumerations (BATCH MODE).

        Args:
            enumerated_groups: EnumeratedGroupNodes instance
            min_confidence: Not used (for API compatibility), kept for consistency

        Returns:
            filtered_enumerated_groups: Filtered EnumeratedGroupNodes instance
        """
        if len(enumerated_groups) == 0:
            return enumerated_groups

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[LLM Filter] GroupNode Enumerations - BEFORE Filtering")
            print(f"{'='*70}")
            print(f"Total: {len(enumerated_groups)} enumerations\n")

            # Show all enumerations before filtering
            for idx, enum_data in enumerate(enumerated_groups.all_enumerations, 1):
                room = enum_data['room_caption']
                objects = enum_data['caption_combo']
                prob = enum_data['probability']
                group_idx = enum_data['group_idx']
                num_objects = len(objects)

                # Show which GroupNode this belongs to
                print(f"{idx:3d}. [{room:12s}] Group#{group_idx} ({num_objects} objs) {objects} (p={prob:.4f})")

        original_count = len(enumerated_groups.all_enumerations)

        # Separate items: 100% certain vs uncertain
        certain_indices = set()  # probability = 1.0, skip LLM
        uncertain_enums = []     # need LLM filtering
        uncertain_indices_map = {}  # maps uncertain enum index to original index

        for idx, enum_data in enumerate(enumerated_groups.all_enumerations):
            if enum_data['probability'] >= 0.9999:  # 100% certain (use 0.9999 to handle float precision)
                certain_indices.add(idx)
            else:
                uncertain_indices_map[len(uncertain_enums)] = idx
                uncertain_enums.append(enum_data)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[LLM Filter] Filtering Strategy")
            print(f"{'='*70}")
            print(f"Total: {original_count} enumerations")
            print(f"  - 100% certain (skip LLM): {len(certain_indices)}")
            print(f"  - Uncertain (send to LLM): {len(uncertain_enums)}")

        # If no uncertain items, keep everything
        if len(uncertain_enums) == 0:
            if self.verbose:
                print(f"\n✓ All enumerations are 100% certain, no LLM filtering needed")
            self.stats['group_kept'] += original_count
            return enumerated_groups

        # Build batch prompt with only uncertain enumerations
        prompt = self._build_batch_group_filter_prompt(uncertain_enums)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[LLM Filter] Calling LLM for {len(uncertain_enums)} uncertain items...")
            print(f"{'='*70}")

        # Call LLM once for uncertain enumerations
        response = self.get_llm_response(prompt)

        if self.verbose:
            print(f"\n[LLM Response]:")
            print(f"{response}\n")

        # Parse batch response (returns indices into uncertain_enums)
        plausible_uncertain_indices = self._parse_batch_filter_response(response, len(uncertain_enums))

        # Map back to original indices
        plausible_indices = certain_indices.copy()  # All certain items are plausible
        for uncertain_idx in plausible_uncertain_indices:
            original_idx = uncertain_indices_map[uncertain_idx]
            plausible_indices.add(original_idx)

        if self.verbose:
            print(f"{'='*70}")
            print(f"[LLM Filter] Filtering Results")
            print(f"{'='*70}")

        # Filter based on LLM response
        kept = []
        filtered_out = []

        for idx, enum_data in enumerate(enumerated_groups.all_enumerations):
            if idx in plausible_indices:
                kept.append(enum_data)
                self.stats['group_kept'] += 1
            else:
                filtered_out.append((idx, enum_data))
                self.stats['group_filtered'] += 1

        # Print filtered items
        if self.verbose:
            print(f"\n❌ FILTERED OUT ({len(filtered_out)} items):")
            for idx, enum_data in filtered_out:
                room = enum_data['room_caption']
                objects = enum_data['caption_combo']
                prob = enum_data['probability']
                group_idx = enum_data['group_idx']
                num_objects = len(objects)
                print(f"  {idx+1:3d}. [{room:12s}] Group#{group_idx} ({num_objects} objs) {objects} (p={prob:.4f})")

        # Update enumerated_groups
        enumerated_groups.all_enumerations = kept
        enumerated_groups._sort()

        # Rebuild indices
        enumerated_groups.by_room = {}
        enumerated_groups.by_group = {}

        for enum_data in kept:
            room_caption = enum_data['room_caption']
            group_key = (room_caption, enum_data['group_idx'])

            if room_caption not in enumerated_groups.by_room:
                enumerated_groups.by_room[room_caption] = []
            enumerated_groups.by_room[room_caption].append(enum_data)

            if group_key not in enumerated_groups.by_group:
                enumerated_groups.by_group[group_key] = []
            enumerated_groups.by_group[group_key].append(enum_data['group_node'])

        if self.verbose:
            print(f"  Result: Kept {len(kept)}/{original_count} ({len(kept)/original_count*100:.1f}%)")
            
        return enumerated_groups

    def filter_room_combinations(self, room_combinations, min_confidence=0.5):
        """
        Filter implausible room-level GroupNode combinations (BATCH MODE).

        Args:
            room_combinations: RoomGroupCombinations instance
            min_confidence: Not used (for API compatibility)

        Returns:
            filtered_room_combinations: Filtered RoomGroupCombinations instance
        """
        if len(room_combinations) == 0:
            return room_combinations

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[LLM Filter] Room Combinations - BEFORE Filtering")
            print(f"{'='*70}")
            print(f"Total: {len(room_combinations)} combinations\n")

            # Show all combinations before filtering
            for idx, combo_data in enumerate(room_combinations.all_combinations, 1):
                room = combo_data['room_caption']
                prob = combo_data['probability']
                num_groups = combo_data['num_groups']

                print(f"{idx:3d}. [{room:12s}] (p={prob:.6f}, {num_groups} groups)")
                for gidx, group in enumerate(combo_data['group_nodes'], 1):
                    captions = [n.caption for n in group.nodes]
                    print(f"     Group {gidx} ({len(captions)} objs): {captions}")

        original_count = len(room_combinations.all_combinations)

        # Separate items: 100% certain vs uncertain
        certain_indices = set()  # probability = 1.0, skip LLM
        uncertain_combos = []     # need LLM filtering
        uncertain_indices_map = {}  # maps uncertain combo index to original index

        for idx, combo_data in enumerate(room_combinations.all_combinations):
            if combo_data['probability'] >= 0.9999:  # 100% certain
                certain_indices.add(idx)
            else:
                uncertain_indices_map[len(uncertain_combos)] = idx
                uncertain_combos.append(combo_data)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[LLM Filter] Filtering Strategy")
            print(f"{'='*70}")
            print(f"Total: {original_count} combinations")
            print(f"  - 100% certain (skip LLM): {len(certain_indices)}")
            print(f"  - Uncertain (send to LLM): {len(uncertain_combos)}")

        # If no uncertain items, keep everything
        if len(uncertain_combos) == 0:
            if self.verbose:
                print(f"\n✓ All combinations are 100% certain, no LLM filtering needed")
            self.stats['room_kept'] += original_count
            return room_combinations

        # Build batch prompt with only uncertain combinations
        prompt = self._build_batch_room_filter_prompt(uncertain_combos)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[LLM Filter] Calling LLM for {len(uncertain_combos)} uncertain items...")
            print(f"{'='*70}")

        # Call LLM once for uncertain combinations
        response = self.get_llm_response(prompt)

        if self.verbose:
            print(f"\n[LLM Response]:")
            print(f"{response}\n")

        # Parse batch response (returns indices into uncertain_combos)
        plausible_uncertain_indices = self._parse_batch_filter_response(response, len(uncertain_combos))

        # Map back to original indices
        plausible_indices = certain_indices.copy()  # All certain items are plausible
        for uncertain_idx in plausible_uncertain_indices:
            original_idx = uncertain_indices_map[uncertain_idx]
            plausible_indices.add(original_idx)

        if self.verbose:
            print(f"{'='*70}")
            print(f"[LLM Filter] Filtering Results")
            print(f"{'='*70}")

        # Filter based on LLM response
        kept = []
        filtered_out = []

        for idx, combo_data in enumerate(room_combinations.all_combinations):
            if idx in plausible_indices:
                kept.append(combo_data)
                self.stats['room_kept'] += 1
            else:
                filtered_out.append((idx, combo_data))
                self.stats['room_filtered'] += 1

        # Update room_combinations
        room_combinations.all_combinations = kept
        room_combinations._sort()

        # Rebuild index
        room_combinations.by_room = {}
        for combo_data in kept:
            room_caption = combo_data['room_caption']
            if room_caption not in room_combinations.by_room:
                room_combinations.by_room[room_caption] = []
            room_combinations.by_room[room_caption].append(combo_data)

        if self.verbose:
            print(f"  Result: Kept {len(kept)}/{original_count} ({len(kept)/original_count*100:.1f}%)")
        return room_combinations

    def _check_group_plausibility(self, room_caption: str, objects: List[str]) -> Tuple[bool, str]:
        """
        Check if a group of objects is plausible in the given room.

        Args:
            room_caption: Room type (e.g., 'bedroom')
            objects: List of object captions (e.g., ['bed', 'table', 'lamp'])

        Returns:
            (is_plausible, reason)
        """
        # Build prompt
        objects_str = ", ".join(objects)
        prompt = self.group_filter_template.format(
            room=room_caption,
            objects=objects_str
        )

        # Call LLM
        response = self.get_llm_response(prompt)

        # Parse response
        is_plausible, reason = self._parse_response(response)

        return is_plausible, reason

    def _check_room_combination_plausibility(self,
                                            room_caption: str,
                                            group_captions: List[List[str]]) -> Tuple[bool, str]:
        """
        Check if a room configuration with multiple groups is plausible.

        Args:
            room_caption: Room type
            group_captions: List of groups, each group is a list of object captions

        Returns:
            (is_plausible, reason)
        """
        # Build groups text
        groups_text = ""
        for idx, group in enumerate(group_captions, 1):
            objects_str = ", ".join(group)
            groups_text += f"Group {idx}: {objects_str}\n"

        # Build prompt
        prompt = self.room_filter_template.format(
            room=room_caption,
            groups=groups_text.strip()
        )

        # Call LLM
        response = self.get_llm_response(prompt)

        # Parse response
        is_plausible, reason = self._parse_response(response)

        return is_plausible, reason

    def _build_batch_group_filter_prompt(self, all_enumerations: List[Dict]) -> str:
        """
        Build batch prompt for filtering multiple GroupNode enumerations at once.

        Args:
            all_enumerations: List of enumeration dicts

        Returns:
            Prompt string for batch filtering
        """
        prompt = """You are evaluating whether object groups in indoor scenes are plausible.

Below is a list of object groups, each with a room type and objects found together.
For each group, determine if the combination is reasonable.

Return your answer as a JSON array with the indices (1-indexed) of ONLY the PLAUSIBLE groups.
Format: {"plausible": [1, 3, 5, ...]}

If all groups are plausible, return all indices. If none are plausible, return an empty array.

Groups to evaluate:
"""

        for idx, enum_data in enumerate(all_enumerations, 1):
            room = enum_data['room_caption']
            objects = enum_data['caption_combo']
            objects_str = ", ".join(objects)
            prompt += f"\n{idx}. Room: {room}, Objects: {objects_str}"

        prompt += "\n\nYour answer (JSON only):"
        return prompt

    def _build_batch_room_filter_prompt(self, all_combinations: List[Dict]) -> str:
        """
        Build batch prompt for filtering multiple room combinations at once.

        Args:
            all_combinations: List of combination dicts

        Returns:
            Prompt string for batch filtering
        """
        prompt = """You are evaluating whether room configurations are plausible.

Below is a list of room configurations, each containing multiple object groups.
For each configuration, determine if the overall setup is reasonable for that room type.

Return your answer as a JSON array with the indices (1-indexed) of ONLY the PLAUSIBLE configurations.
Format: {"plausible": [1, 2, 4, ...]}

If all configurations are plausible, return all indices. If none are plausible, return an empty array.

Configurations to evaluate:
"""

        for idx, combo_data in enumerate(all_combinations, 1):
            room = combo_data['room_caption']
            prompt += f"\n{idx}. Room: {room}"

            # Add groups
            for gidx, group in enumerate(combo_data['group_nodes'], 1):
                captions = [n.caption for n in group.nodes]
                objects_str = ", ".join(captions)
                prompt += f"\n   Group {gidx}: {objects_str}"

        prompt += "\n\nYour answer (JSON only):"
        return prompt

    def _parse_batch_filter_response(self, response: str, total_count: int) -> set:
        """
        Parse LLM batch filter response to extract plausible indices.

        Expected format: {"plausible": [1, 3, 5, ...]}

        Args:
            response: LLM response string
            total_count: Total number of items evaluated

        Returns:
            Set of 0-indexed plausible item indices
        """
        # Try to extract JSON
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^}]*"plausible"[^}]*\}', response, re.IGNORECASE | re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                plausible_list = data.get('plausible', [])

                # Convert 1-indexed to 0-indexed
                plausible_indices = set(idx - 1 for idx in plausible_list if 1 <= idx <= total_count)
                return plausible_indices
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            if self.verbose:
                print(f"  Warning: Could not parse JSON response: {e}")
                print(f"  Response: {response[:200]}...")

        # Fallback: try to extract numbers from response
        try:
            # Look for patterns like "1, 3, 5" or "[1, 3, 5]"
            numbers = re.findall(r'\d+', response)
            plausible_list = [int(n) for n in numbers if 1 <= int(n) <= total_count]
            plausible_indices = set(idx - 1 for idx in plausible_list)
            if plausible_indices and self.verbose:
                print(f"  Fallback: Extracted indices {sorted(plausible_indices)}")
            return plausible_indices
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Fallback parsing failed: {e}")

        # Last resort: assume all are plausible
        if self.verbose:
            print(f"  Warning: Could not parse response, assuming all {total_count} items are plausible")
        return set(range(total_count))

    def _parse_response(self, response: str) -> Tuple[bool, str]:
        """
        Parse LLM response to extract plausibility judgment.

        Expected format:
        {"plausible": true/false, "reason": "explanation"}

        Args:
            response: Raw LLM response

        Returns:
            (is_plausible, reason)
        """
        # Try JSON parsing first
        try:
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{[^}]*"plausible"[^}]*\}', response, re.IGNORECASE | re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                is_plausible = data.get('plausible', True)
                reason = data.get('reason', '')
                return (is_plausible, reason)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: simple text parsing
        response_lower = response.lower()

        # Check for negative keywords
        if any(word in response_lower for word in ['implausible', 'not plausible', 'unreasonable', 'unlikely', 'false']):
            # Try to extract reason
            if ':' in response:
                reason = response.split(':', 1)[1].strip()
            else:
                reason = response.strip()
            return (False, reason)

        # Default: assume plausible
        return (True, "")

    def get_stats(self) -> Dict:
        """Get filtering statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset filtering statistics."""
        self.stats = {
            'group_filtered': 0,
            'group_kept': 0,
            'room_filtered': 0,
            'room_kept': 0,
        }

    def print_stats(self):
        """Print filtering statistics."""
        print("\n" + "="*60)
        print("LLM Filter Statistics")
        print("="*60)
        print(f"GroupNode Enumerations:")
        print(f"  Kept: {self.stats['group_kept']}")
        print(f"  Filtered: {self.stats['group_filtered']}")

        total_groups = self.stats['group_kept'] + self.stats['group_filtered']
        if total_groups > 0:
            print(f"  Filter rate: {self.stats['group_filtered']/total_groups*100:.1f}%")

        print(f"\nRoom Combinations:")
        print(f"  Kept: {self.stats['room_kept']}")
        print(f"  Filtered: {self.stats['room_filtered']}")

        total_rooms = self.stats['room_kept'] + self.stats['room_filtered']
        if total_rooms > 0:
            print(f"  Filter rate: {self.stats['room_filtered']/total_rooms*100:.1f}%")

        print("="*60)
