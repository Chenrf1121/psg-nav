"""
Staircase Climbing Module

This module handles the entire staircase climbing process:
1. Detect staircase in view
2. Navigate to staircase
3. Align with staircase using depth map
4. Climb stairs while monitoring safety

Author: Claude Code
Date: 2025-12-17
"""

import numpy as np
import cv2
from enum import Enum
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class StaircaseClimbState(Enum):
    """States for staircase climbing state machine"""
    IDLE = 0
    NAVIGATING_TO_STAIRCASE = 1
    ALIGNING_WITH_STAIRCASE = 2
    CLIMBING = 3
    COMPLETED = 4
    FAILED = 5


class StaircaseClimber:
    """
    Handles staircase detection, navigation, and climbing.

    Uses depth map analysis to safely climb stairs by:
    - Detecting staircase pattern in depth image
    - Aligning robot to face staircase directly
    - Climbing while monitoring depth gradient
    """

    def __init__(self, agent):
        """
        Initialize StaircaseClimber.

        Args:
            agent: Reference to main PSG_Nav_Agent instance
        """
        self.agent = agent
        self.state = StaircaseClimbState.IDLE
        self.StaircaseClimbState = StaircaseClimbState  # Make accessible to main agent

        # Staircase detection parameters
        self.staircase_in_view = False
        self.staircase_center_x = None  # Horizontal position of staircase in image
        self.staircase_depth_profile = None  # Depth gradient pattern
        self.center_x_history = []  # History for smoothing center position
        self.center_x_history_size = 3  # Use last 3 frames for smoothing

        # Navigation parameters
        self.target_staircase = None  # Target staircase to navigate to
        self.navigation_goal_reached = False

        # Alignment parameters
        self.alignment_threshold = 100  # pixels from image center (increased from 50 to avoid oscillation)
        self.aligned_frames = 0  # Count consecutive aligned frames
        self.min_aligned_frames = 3  # Need stable alignment before climbing

        # Climbing parameters
        self.climbing_started = False
        self.climbing_steps = 0
        self.max_climbing_steps = 25  # Maximum steps to attempt climbing (reduced from 50)
        self.stuck_threshold = 5  # Steps without progress = stuck
        self.stuck_count = 0

        # Depth analysis parameters
        self.depth_gradient_threshold = 0.002  # meters per vertical pixel (reduced from 0.01 to be more sensitive)
        self.min_staircase_width = 100  # pixels
        self.staircase_depth_range = (0.5, 3.0)  # meters (min, max)

        # Safety parameters
        self.last_depth_mean = None
        self.depth_increase_threshold = 0.3  # meters - detect approaching wall

        # Exit detection parameters
        self.no_staircase_frames = 0  # Count frames without staircase detected
        self.exit_threshold = 3  # Consecutive frames without staircase = exited (reduced from 5)

        # Visualization parameters
        self.enable_visualization = False  # Enable saving detection visualizations
        self.viz_counter = 0  # Counter for visualization filenames
        self.viz_dir = "staircase_viz"  # Directory to save visualizations
        if self.enable_visualization:
            os.makedirs(self.viz_dir, exist_ok=True)

    def reset(self):
        """Reset climber state for new episode or after completion."""
        self.state = StaircaseClimbState.IDLE
        self.staircase_in_view = False
        self.staircase_center_x = None
        self.staircase_depth_profile = None
        self.center_x_history = []
        self.target_staircase = None
        self.navigation_goal_reached = False
        self.aligned_frames = 0
        self.climbing_started = False
        self.climbing_steps = 0
        self.stuck_count = 0
        self.last_depth_mean = None
        self.no_staircase_frames = 0

    def should_climb_staircase(self):
        """
        Decide if robot should attempt staircase climbing.

        Returns:
            bool: True if staircase climbing should be initiated
        """
        # Check if staircase was detected
        if len(self.agent.staircase_locations) == 0:
            return False

        # Check if we're not already climbing
        if self.state not in [StaircaseClimbState.IDLE, StaircaseClimbState.COMPLETED]:
            return False

        # Check if staircase is close enough to attempt
        staircase = self.agent.staircase_locations[0]
        agent_pos = np.array([self.agent.full_pose[0].cpu().numpy(),
                             self.agent.full_pose[1].cpu().numpy()])
        staircase_gps = np.array(staircase['gps'])
        distance = np.linalg.norm(agent_pos - staircase_gps)

        # If staircase is within 3 meters, consider climbing
        if distance < 3.0:
            return True

        return False

    def start_climbing(self):
        """Initiate staircase climbing sequence."""
        if len(self.agent.staircase_locations) == 0:
            print("[StaircaseClimber] No staircase detected, cannot start climbing")
            return False

        self.target_staircase = self.agent.staircase_locations[0]
        self.state = StaircaseClimbState.NAVIGATING_TO_STAIRCASE
        print(f"[StaircaseClimber] Starting climb sequence for staircase at "
              f"GPS: {self.target_staircase['gps']}, "
              f"Confidence: {self.target_staircase['confidence']:.2f}")
        return True

    def visualize_detection(self, depth, region_depth, vertical_gradient, gradients_per_strip,
                           positive_strips, negative_strips, detected, bottom_third_start, center_w_start):
        """
        Create visualization of staircase detection process.

        Args:
            depth: Original depth map
            region_depth: Bottom 1/3 region analyzed
            vertical_gradient: Computed gradient
            gradients_per_strip: List of gradient values per strip
            positive_strips: Number of positive strips
            negative_strips: Number of negative strips
            detected: Whether staircase was detected
            bottom_third_start: Row where bottom 1/3 starts
            center_w_start: Column where center region starts
        """
        self.viz_counter += 1

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Original depth map with analysis region highlighted
        ax = axes[0, 0]
        depth_vis = depth.squeeze() if depth.ndim == 3 else depth
        im1 = ax.imshow(depth_vis, cmap='viridis', vmin=0, vmax=10)
        ax.set_title('Depth Map (Full)')

        # Highlight bottom 1/3 region
        h, w = depth_vis.shape
        rect = patches.Rectangle((center_w_start, bottom_third_start),
                                 w//2, h - bottom_third_start,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(center_w_start + 10, bottom_third_start + 20,
                'Analysis Region\n(Bottom 1/3)', color='red', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.colorbar(im1, ax=ax, label='Depth (m)')

        # 2. Bottom 1/3 region (analyzed region)
        ax = axes[0, 1]
        im2 = ax.imshow(region_depth, cmap='viridis', vmin=0, vmax=10)
        ax.set_title('Bottom 1/3 Region (Analyzed)')
        plt.colorbar(im2, ax=ax, label='Depth (m)')

        # 3. Gradient map
        ax = axes[1, 0]
        im3 = ax.imshow(vertical_gradient, cmap='RdBu', vmin=-0.05, vmax=0.05)
        ax.set_title('Vertical Gradient (positive=depth increasing)')

        # Draw strip boundaries
        strip_height = max(region_depth.shape[0] // 5, 5)
        for i, grad in enumerate(gradients_per_strip):
            y_pos = i * strip_height // 2
            color = 'green' if abs(grad) > self.depth_gradient_threshold else 'gray'
            ax.axhline(y=y_pos, color=color, linestyle='--', alpha=0.5, linewidth=1)
            ax.text(5, y_pos + 5, f'Strip {i}: {grad:.4f}',
                   color='white', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

        plt.colorbar(im3, ax=ax, label='Gradient (m/pixel)')

        # 4. Strip gradients bar chart
        ax = axes[1, 1]
        colors = ['green' if g > self.depth_gradient_threshold else
                 'red' if g < -self.depth_gradient_threshold else 'gray'
                 for g in gradients_per_strip]
        ax.bar(range(len(gradients_per_strip)), gradients_per_strip, color=colors)
        ax.axhline(y=self.depth_gradient_threshold, color='green', linestyle='--',
                  label=f'Threshold: {self.depth_gradient_threshold}')
        ax.axhline(y=-self.depth_gradient_threshold, color='red', linestyle='--')
        ax.set_xlabel('Strip Index')
        ax.set_ylabel('Gradient (m/pixel)')
        ax.set_title(f'Strip Gradients\nPositive: {positive_strips}, Negative: {negative_strips}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Overall result
        result_text = f"DETECTED: {detected}\n"
        result_text += f"Positive strips: {positive_strips} (need ≥2)\n"
        result_text += f"Negative strips: {negative_strips} (need ≥2)"
        fig.text(0.5, 0.02, result_text, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgreen' if detected else 'lightcoral', alpha=0.8))

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Save figure
        filename = os.path.join(self.viz_dir, f'detection_{self.viz_counter:04d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"[Visualization] Saved to {filename}")

    def detect_staircase_in_view(self, observations):
        """
        Detect if staircase is visible in current view using depth map.

        Analyzes depth image to find characteristic staircase pattern:
        - Vertical gradient (depth increases going down in image)
        - Horizontal consistency (similar pattern across width)
        - Appropriate depth range

        Args:
            observations: Habitat observations with 'depth' key

        Returns:
            bool: True if staircase detected in view
        """
        depth = observations['depth']
        h, w = depth.shape[:2]

        # Focus on bottom 1/3 of image (where stairs should be)
        # Top 2/3 might contain distant objects, walls, ceiling
        bottom_third_start = 2 * h // 3
        bottom_depth = depth[bottom_third_start:, :]

        # Also focus on center region horizontally (stairs should be in front)
        center_w_start = w // 4
        center_w_end = 3 * w // 4
        region_depth = bottom_depth[:, center_w_start:center_w_end]
        if region_depth.ndim == 3:
            region_depth = region_depth.squeeze(-1)  # Remove channel dimension if present

        region_h = region_depth.shape[0]
        # Compute vertical gradient (depth change going down)
        # Stairs have positive gradient (depth increases going down image)
        vertical_gradient = np.diff(region_depth, axis=0)

        # Average gradient across horizontal strips
        strip_height = max(region_h // 5, 5)  # At least 5 pixels per strip
        gradients_per_strip = []
        for i in range(0, region_h - strip_height, strip_height // 2):
            strip = vertical_gradient[i:i+strip_height, :]
            avg_gradient = np.median(strip)
            gradients_per_strip.append(avg_gradient)

        # Staircase has consistent positive OR negative gradient (stairs can go up or down)
        positive_strips = sum(1 for g in gradients_per_strip if g > self.depth_gradient_threshold)
        negative_strips = sum(1 for g in gradients_per_strip if g < -self.depth_gradient_threshold)

        # Determine detection result
        detected = (positive_strips >= 2 or negative_strips >= 2)

        # Create visualization (commented out - not needed)
        if self.enable_visualization:
            self.visualize_detection(depth, region_depth, vertical_gradient, gradients_per_strip,
                                    positive_strips, negative_strips, detected,
                                    bottom_third_start, center_w_start)

        # Staircase has consistent positive OR negative gradient (stairs can go up or down)
        # Require at least 2 strips with consistent gradient (balanced detection)
        if detected:
            # Compute horizontal center of staircase
            # Find column with strongest gradient signal
            column_gradients = np.sum(np.abs(vertical_gradient), axis=0)
            max_gradient_col = np.argmax(column_gradients)
            raw_center_x = center_w_start + max_gradient_col

            # Smooth the center position using moving average to reduce jitter
            self.center_x_history.append(raw_center_x)
            if len(self.center_x_history) > self.center_x_history_size:
                self.center_x_history.pop(0)

            self.staircase_center_x = int(np.mean(self.center_x_history))

            # Store depth profile for analysis
            self.staircase_depth_profile = region_depth[:, max_gradient_col]

            self.staircase_in_view = True
            print(f"[StaircaseClimber] ✓ Staircase detected in view, raw_center_x: {raw_center_x}, "
                  f"smoothed_center_x: {self.staircase_center_x}, positive_strips: {positive_strips}, negative_strips: {negative_strips}")
            return True

        self.staircase_in_view = False
        self.center_x_history = []  # Clear history when not detected
        print(f"[StaircaseClimber] ✗ No staircase detected (need ≥2 consistent strips)")
        return False

    def is_aligned_with_staircase(self, observations):
        """
        Check if robot is properly aligned with staircase.

        Args:
            observations: Habitat observations

        Returns:
            bool: True if aligned (staircase is centered in view)
        """
        if not self.staircase_in_view:
            return False

        depth = observations['depth']
        h, w = depth.shape[:2]
        image_center = w // 2

        # Check if staircase center is close to image center
        offset = abs(self.staircase_center_x - image_center)

        if offset < self.alignment_threshold:
            self.aligned_frames += 1
            if self.aligned_frames >= self.min_aligned_frames:
                return True
        else:
            self.aligned_frames = 0

        return False

    def get_alignment_action(self, observations):
        """
        Get action to align robot with staircase.

        Args:
            observations: Habitat observations

        Returns:
            int: Action to take (2=TURN_LEFT, 3=TURN_RIGHT)
        """
        if not self.staircase_in_view:
            return 3  # Turn right to search

        depth = observations['depth']
        w = depth.shape[1]
        image_center = w // 2

        # Turn to center the staircase
        if self.staircase_center_x < image_center - self.alignment_threshold:
            return 2  # TURN_LEFT
        elif self.staircase_center_x > image_center + self.alignment_threshold:
            return 3  # TURN_RIGHT
        else:
            return 1  # MOVE_FORWARD (aligned)

    def is_still_on_staircase(self, observations):
        """
        Check if agent is still on the staircase.

        Detects staircase exit by checking if the characteristic staircase
        pattern disappears or becomes too uniform (indicating flat ground).

        Args:
            observations: Habitat observations

        Returns:
            bool: True if still on staircase, False if exited
        """
        depth = observations['depth']

        # Check if staircase pattern still visible
        staircase_detected = self.detect_staircase_in_view(observations)

        if not staircase_detected:
            self.no_staircase_frames += 1
            print(f"[StaircaseClimber] No staircase pattern detected ({self.no_staircase_frames}/{self.exit_threshold} frames)")

            if self.no_staircase_frames >= self.exit_threshold:
                print(f"[StaircaseClimber] Staircase pattern lost for {self.exit_threshold} frames - exited staircase")
                return False
        else:
            # Check if gradient pattern indicates flat ground (too many strips with gradient)
            # On stairs: 2-5 strips have gradient
            # On flat ground or after exiting: >5 strips have gradient (uniform slope)
            h, w = depth.shape[:2]
            bottom_third_start = 2 * h // 3
            bottom_depth = depth[bottom_third_start:, :]
            center_w_start = w // 4
            center_w_end = 3 * w // 4
            region_depth = bottom_depth[:, center_w_start:center_w_end]
            if region_depth.ndim == 3:
                region_depth = region_depth.squeeze(-1)

            vertical_gradient = np.diff(region_depth, axis=0)
            region_h = region_depth.shape[0]
            strip_height = max(region_h // 5, 5)

            gradients_per_strip = []
            for i in range(0, region_h - strip_height, strip_height // 2):
                strip = vertical_gradient[i:i+strip_height, :]
                avg_gradient = np.median(strip)
                gradients_per_strip.append(avg_gradient)

            positive_strips = sum(1 for g in gradients_per_strip if g > self.depth_gradient_threshold)
            negative_strips = sum(1 for g in gradients_per_strip if g < -self.depth_gradient_threshold)
            total_gradient_strips = positive_strips + negative_strips

            print(f"[StaircaseExit] Total gradient strips: {total_gradient_strips} (pos: {positive_strips}, neg: {negative_strips})")

            if total_gradient_strips > 5:
                print(f"[StaircaseClimber] Too many gradient strips ({total_gradient_strips} > 5) - likely exited to flat ground")
                return False

            # Reset counter if still on stairs
            self.no_staircase_frames = 0

        # Check for obstacles (sudden depth decrease in center)
        center_depth = depth[depth.shape[0]//3:2*depth.shape[0]//3,
                            depth.shape[1]//3:2*depth.shape[1]//3]
        current_depth_mean = np.median(center_depth[center_depth > 0])

        if self.last_depth_mean is not None:
            depth_change = current_depth_mean - self.last_depth_mean
            # If depth suddenly increases significantly, might have hit wall or obstacle
            if depth_change > self.depth_increase_threshold:
                print(f"[StaircaseClimber] Sudden depth increase: {depth_change:.2f}m, stopping")
                return False

        self.last_depth_mean = current_depth_mean

        return True

    def get_climbing_action(self, observations):
        """
        Get next action for climbing stairs.

        Args:
            observations: Habitat observations

        Returns:
            int: Action to take during climbing, or None if should exit
        """
        # Check if still on staircase
        if not self.is_still_on_staircase(observations):
            print("[StaircaseClimber] Exited staircase")
            self.state = StaircaseClimbState.COMPLETED
            return 0  # STOP

        # Check if still aligned
        if self.staircase_in_view and not self.is_aligned_with_staircase(observations):
            # Need realignment
            print("[StaircaseClimber] Need realignment during climbing")
            return self.get_alignment_action(observations)

        # Move forward to climb
        self.climbing_steps += 1

        # Check termination conditions
        if self.climbing_steps >= self.max_climbing_steps:
            print(f"[StaircaseClimber] Reached max climbing steps ({self.max_climbing_steps})")
            self.state = StaircaseClimbState.COMPLETED
            return 0  # STOP

        return 1  # MOVE_FORWARD

    def step(self, observations):
        """
        Execute one step of staircase climbing state machine.

        Args:
            observations: Habitat observations dict

        Returns:
            dict: Action dict {'action': int} or None if not handling
        """
        if self.state == StaircaseClimbState.IDLE:
            # Not climbing - this shouldn't be called in IDLE state
            print("[StaircaseClimber] Warning: step() called in IDLE state")
            return None

        elif self.state == StaircaseClimbState.NAVIGATING_TO_STAIRCASE:
            # Navigate to staircase position using existing FBE navigation
            # This state is handled by main navigation, not here
            if self.detect_staircase_in_view(observations):
                print("[StaircaseClimber] Staircase in view, transitioning to alignment")
                self.state = StaircaseClimbState.ALIGNING_WITH_STAIRCASE
                # Fall through to alignment logic below
            else:
                return None  # Let main navigation handle this

        if self.state == StaircaseClimbState.ALIGNING_WITH_STAIRCASE:
            # Align robot to face staircase
            if not self.detect_staircase_in_view(observations):
                # Lost sight of staircase
                print("[StaircaseClimber] Lost staircase during alignment")
                self.state = StaircaseClimbState.FAILED
                return None

            if self.is_aligned_with_staircase(observations):
                print("[StaircaseClimber] Aligned! Starting climb")
                self.state = StaircaseClimbState.CLIMBING
                self.climbing_started = True
                return {"action": 1}  # First forward step
            else:
                action = self.get_alignment_action(observations)
                print(f"[StaircaseClimber] Aligning: action={action}")
                return {"action": action}

        elif self.state == StaircaseClimbState.CLIMBING:
            # Execute climbing
            action = self.get_climbing_action(observations)

            if self.state == StaircaseClimbState.COMPLETED:
                # Changed to COMPLETED in get_climbing_action
                print("[StaircaseClimber] Climbing completed!")
                return None

            return {"action": action}

        elif self.state in [StaircaseClimbState.COMPLETED, StaircaseClimbState.FAILED]:
            # Done
            return None

        return None
