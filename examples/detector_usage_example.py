"""
Example usage of the unified object detector

This script demonstrates how to use both GLIP and FastSAM detectors
with the same interface
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add PSG-Nav to path
psgnav_path = Path(__file__).parent.parent
sys.path.insert(0, str(psgnav_path))

from utils.object_detector import create_detector
from configs.detector_config import (
    DETECTOR_TYPE, GLIP_CONFIG, FASTSAM_CONFIG, CATEGORIES_21
)


def visualize_detections(image, bboxes, labels, scores):
    """Visualize detection results"""
    image_vis = image.copy()

    for bbox, label, score in zip(bboxes, labels, scores):
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(image_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label and score
        text = f"{label}: {score:.2f}"
        cv2.putText(image_vis, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_vis


def example_1_basic_usage():
    """Example 1: Basic usage with GLIP"""
    print("\n" + "="*80)
    print("Example 1: Basic Usage with GLIP")
    print("="*80)

    # Create GLIP detector
    detector = create_detector('glip', **GLIP_CONFIG)

    # Load test image
    image_path = 'test_images/living_room.jpg'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect objects
    categories = ['chair', 'table', 'sofa']
    results = detector.detect(image_rgb, categories)

    # Print results
    print(f"\nDetected {len(results['labels'])} objects:")
    for i, (bbox, label, score) in enumerate(zip(
        results['bboxes'], results['labels'], results['scores']
    )):
        print(f"  {i+1}. {label}: score={score:.3f}, bbox={bbox}")

    # Visualize
    image_vis = visualize_detections(image_rgb, results['bboxes'],
                                     results['labels'], results['scores'])
    cv2.imwrite('output_glip.jpg', cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
    print("\nVisualization saved to output_glip.jpg")


def example_2_fastsam_usage():
    """Example 2: Using FastSAM detector"""
    print("\n" + "="*80)
    print("Example 2: Using FastSAM Detector")
    print("="*80)

    # Create FastSAM detector
    detector = create_detector('fastsam', **FASTSAM_CONFIG)

    # Load test image
    image_path = 'test_images/living_room.jpg'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect objects
    categories = ['chair', 'table', 'sofa']
    results = detector.detect(image_rgb, categories)

    # Print results
    print(f"\nDetected {len(results['labels'])} objects:")
    for i, (bbox, label, score) in enumerate(zip(
        results['bboxes'], results['labels'], results['scores']
    )):
        print(f"  {i+1}. {label}: score={score:.3f}, bbox={bbox}")

    # Visualize
    image_vis = visualize_detections(image_rgb, results['bboxes'],
                                     results['labels'], results['scores'])
    cv2.imwrite('output_fastsam.jpg', cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
    print("\nVisualization saved to output_fastsam.jpg")


def example_3_all_categories():
    """Example 3: Detect all 21 categories"""
    print("\n" + "="*80)
    print("Example 3: Detect All 21 Categories")
    print("="*80)

    # Create detector based on config
    if DETECTOR_TYPE == 'glip':
        detector = create_detector('glip', **GLIP_CONFIG)
    else:
        detector = create_detector('fastsam', **FASTSAM_CONFIG)

    print(f"Using detector: {DETECTOR_TYPE.upper()}")

    # Load test image
    image_path = 'test_images/bedroom.jpg'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect all categories
    results = detector.detect(image_rgb, CATEGORIES_21)

    # Print category summary
    print(f"\nTotal detections: {len(results['labels'])}")
    print("\nDetections by category:")
    for category in CATEGORIES_21:
        count = results['labels'].count(category)
        if count > 0:
            print(f"  {category}: {count}")

    # Visualize
    image_vis = visualize_detections(image_rgb, results['bboxes'],
                                     results['labels'], results['scores'])
    cv2.imwrite(f'output_{DETECTOR_TYPE}_all.jpg',
                cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
    print(f"\nVisualization saved to output_{DETECTOR_TYPE}_all.jpg")


def example_4_speed_comparison():
    """Example 4: Speed comparison between GLIP and FastSAM"""
    print("\n" + "="*80)
    print("Example 4: Speed Comparison")
    print("="*80)

    import time

    # Load test image
    image_path = 'test_images/living_room.jpg'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    categories = ['chair', 'table', 'sofa', 'bed']

    # Test GLIP
    print("\nTesting GLIP detector...")
    detector_glip = create_detector('glip', **GLIP_CONFIG)

    times_glip = []
    for i in range(5):
        start = time.time()
        results = detector_glip.detect(image_rgb, categories)
        elapsed = time.time() - start
        times_glip.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s, detections: {len(results['labels'])}")

    avg_glip = np.mean(times_glip[1:])  # Skip first run (warmup)
    print(f"Average GLIP time (excluding warmup): {avg_glip:.3f}s")

    # Test FastSAM
    print("\nTesting FastSAM detector...")
    detector_fastsam = create_detector('fastsam', **FASTSAM_CONFIG)

    times_fastsam = []
    for i in range(5):
        start = time.time()
        results = detector_fastsam.detect(image_rgb, categories)
        elapsed = time.time() - start
        times_fastsam.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s, detections: {len(results['labels'])}")

    avg_fastsam = np.mean(times_fastsam[1:])
    print(f"Average FastSAM time (excluding warmup): {avg_fastsam:.3f}s")

    # Summary
    print("\n" + "="*80)
    print("Speed Comparison Summary:")
    print("="*80)
    print(f"GLIP:    {avg_glip:.3f}s")
    print(f"FastSAM: {avg_fastsam:.3f}s")
    print(f"Speedup: {avg_glip / avg_fastsam:.2f}x")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("Unified Object Detector Examples")
    print("="*80)
    print(f"Current detector type: {DETECTOR_TYPE.upper()}")
    print("="*80)

    # Run examples
    try:
        example_1_basic_usage()
    except Exception as e:
        print(f"Error in example 1: {e}")

    try:
        example_2_fastsam_usage()
    except Exception as e:
        print(f"Error in example 2: {e}")

    try:
        example_3_all_categories()
    except Exception as e:
        print(f"Error in example 3: {e}")

    try:
        example_4_speed_comparison()
    except Exception as e:
        print(f"Error in example 4: {e}")

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == '__main__':
    main()
