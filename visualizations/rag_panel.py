"""
RAG visualization panel for episode videos

This module creates a compact visualization of RAG state to be embedded
in the main visualization video.
"""

import cv2
import numpy as np
from pathlib import Path


def create_rag_panel(agent, panel_size=(250, 400)):
    """
    Create a compact RAG information panel showing:
    - RAG status (enabled/disabled)
    - Current verification state
    - Knowledge base statistics
    - Recent detections
    - Comparison document (if available)

    Args:
        agent: The agent instance with RAG manager
        panel_size: (width, height) of the panel

    Returns:
        RGB image array of size panel_size
    """
    width, height = panel_size
    panel = np.full((height, width, 3), 255, dtype=np.uint8)  # White background (RGB format)

    # Colors (BGR values to match the visualizer.py convention)
    # visualize_image is RGB format but uses BGR color values
    # because cv2 functions expect BGR, and the final image is converted at the end
    COLOR_TEXT = (0, 0, 0)  # Black
    COLOR_HEADER = (150, 50, 50)  # Dark blue (BGR: B=150, G=50, R=50)
    COLOR_SUCCESS = (0, 150, 0)  # Green (BGR: B=0, G=150, R=0)
    COLOR_WARNING = (0, 100, 200)  # Orange (BGR: B=0, G=100, R=200)
    COLOR_ERROR = (0, 0, 200)  # Red (BGR: B=0, G=0, R=200)
    COLOR_BORDER = (128, 128, 128)  # Gray

    # Draw border
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), COLOR_BORDER, 1)

    # Title
    y_offset = 20
    cv2.putText(panel, "RAG Status", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, COLOR_HEADER, 2, cv2.LINE_AA)
    cv2.line(panel, (10, y_offset + 5), (width - 10, y_offset + 5), COLOR_BORDER, 1)
    y_offset += 25

    # Check if RAG is available
    has_rag_manager = hasattr(agent, 'rag_manager') and agent.rag_manager is not None
    has_rag_detection = hasattr(agent, 'rag_best_detection')

    if not has_rag_manager and not has_rag_detection:
        cv2.putText(panel, "RAG: Disabled", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLOR_ERROR, 1, cv2.LINE_AA)
        return panel

    # RAG Statistics (if manager is available)
    if has_rag_manager:
        stats = agent.rag_manager.get_statistics()
        total_docs = stats['total_documents']
        n_categories = len(stats['categories'])

        # Show total documents
        cv2.putText(panel, f"Total Docs: {total_docs}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        y_offset += 18

        cv2.putText(panel, f"Categories: {n_categories}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        y_offset += 18

        # Current goal
        if hasattr(agent, 'obj_goal'):
            goal_docs = stats['docs_per_category'].get(agent.obj_goal, 0)
            cv2.putText(panel, f"Goal: {agent.obj_goal}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HEADER, 1, cv2.LINE_AA)
            y_offset += 18

            cv2.putText(panel, f"  Docs: {goal_docs}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
            y_offset += 20
    else:
        # No RAG manager, just show goal
        if hasattr(agent, 'obj_goal'):
            cv2.putText(panel, f"Goal: {agent.obj_goal}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HEADER, 1, cv2.LINE_AA)
            y_offset += 20

    # Show RAG check status
    if hasattr(agent, 'rag_check_enabled'):
        status_text = "RAG Check: Enabled" if agent.rag_check_enabled else "RAG Check: Disabled"
        status_color = COLOR_SUCCESS if agent.rag_check_enabled else COLOR_WARNING
        cv2.putText(panel, status_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
        y_offset += 18

    # Show sliding window status
    if hasattr(agent, 'rag_sliding_window'):
        window_text = f"Window: {len(agent.rag_sliding_window)}/{agent.rag_sliding_window_size} detections"
        cv2.putText(panel, window_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        y_offset += 20

    # Show current detection state
    cv2.line(panel, (10, y_offset), (width - 10, y_offset), COLOR_BORDER, 1)
    y_offset += 15

    if hasattr(agent, 'rag_latest_goal_detection') and agent.rag_latest_goal_detection is not None:
        det = agent.rag_latest_goal_detection

        cv2.putText(panel, "Latest Goal Detection:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HEADER, 1, cv2.LINE_AA)
        y_offset += 18

        # Caption
        caption_text = f"  '{det['caption']}'"
        if len(caption_text) > 30:
            caption_text = caption_text[:27] + "..."
        cv2.putText(panel, caption_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        y_offset += 15

        # Confidence
        conf = det['confidence']
        conf_color = COLOR_SUCCESS if conf > 0.7 else (COLOR_WARNING if conf > 0.5 else COLOR_ERROR)
        cv2.putText(panel, f"  Conf: {conf:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1, cv2.LINE_AA)
        y_offset += 15

        # Distance
        cv2.putText(panel, f"  Dist: {det['distance']:.2f}m", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        y_offset += 15

        # Verified status
        verified_text = "Verified" if det['verified'] else "Not Verified"
        verified_color = COLOR_SUCCESS if det['verified'] else COLOR_WARNING
        cv2.putText(panel, f"  {verified_text}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, verified_color, 1, cv2.LINE_AA)
        y_offset += 20

        # Show crop thumbnail if available
        if 'crop' in det and det['crop'] is not None and det['crop'].size > 0:
            crop = det['crop']
            # Resize crop to fit in panel
            crop_height = 60
            aspect_ratio = crop.shape[1] / crop.shape[0]
            crop_width = int(crop_height * aspect_ratio)

            if crop_width > width - 20:
                crop_width = width - 20
                crop_height = int(crop_width / aspect_ratio)

            crop_resized = cv2.resize(crop, (crop_width, crop_height))

            # Crop is in RGB format (from observations["rgb"])
            # Keep it as RGB to match panel format (both are RGB format images)
            # The whole visualize_image will be converted to BGR at the end

            # Place crop in panel
            crop_y = y_offset
            crop_x = (width - crop_width) // 2

            if crop_y + crop_height < height - 10:
                panel[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width] = crop_resized
                cv2.rectangle(panel, (crop_x - 1, crop_y - 1),
                             (crop_x + crop_width, crop_y + crop_height),
                             COLOR_BORDER, 1)
                y_offset += crop_height + 10

    else:
        cv2.putText(panel, "No detection yet", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARNING, 1, cv2.LINE_AA)
        y_offset += 20

    # Show comparison document if RAG verification was performed
    if hasattr(agent, 'rag_comparison_doc') and agent.rag_comparison_doc is not None:
        comp_doc = agent.rag_comparison_doc

        # Draw separator
        cv2.line(panel, (10, y_offset), (width - 10, y_offset), COLOR_BORDER, 1)
        y_offset += 15

        cv2.putText(panel, "RAG Comparison:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HEADER, 1, cv2.LINE_AA)
        y_offset += 18

        # Show document caption
        doc_caption = f"  '{comp_doc.caption}'"
        if len(doc_caption) > 30:
            doc_caption = doc_caption[:27] + "..."
        cv2.putText(panel, doc_caption, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        y_offset += 15

        # Show document confidence
        cv2.putText(panel, f"  Conf: {comp_doc.confidence:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        y_offset += 15

        # Show document episode ID
        episode_text = f"  Episode: {comp_doc.episode_id}"
        if len(episode_text) > 30:
            episode_text = episode_text[:27] + "..."
        cv2.putText(panel, episode_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        y_offset += 20

        # Show document crop thumbnail
        if comp_doc.crop is not None and comp_doc.crop.size > 0:
            doc_crop = comp_doc.crop
            # Resize crop to fit in panel
            doc_crop_height = 60
            doc_aspect_ratio = doc_crop.shape[1] / doc_crop.shape[0]
            doc_crop_width = int(doc_crop_height * doc_aspect_ratio)

            if doc_crop_width > width - 20:
                doc_crop_width = width - 20
                doc_crop_height = int(doc_crop_width / doc_aspect_ratio)

            doc_crop_resized = cv2.resize(doc_crop, (doc_crop_width, doc_crop_height))

            # Place document crop in panel
            doc_crop_y = y_offset
            doc_crop_x = (width - doc_crop_width) // 2

            if doc_crop_y + doc_crop_height < height - 10:
                panel[doc_crop_y:doc_crop_y + doc_crop_height, doc_crop_x:doc_crop_x + doc_crop_width] = doc_crop_resized
                cv2.rectangle(panel, (doc_crop_x - 1, doc_crop_y - 1),
                             (doc_crop_x + doc_crop_width, doc_crop_y + doc_crop_height),
                             COLOR_SUCCESS, 2)  # Green border for comparison doc
                y_offset += doc_crop_height + 10

    # Show top categories in knowledge base (if space available and manager exists)
    if has_rag_manager and y_offset < height - 60 and n_categories > 0:
        cv2.line(panel, (10, y_offset), (width - 10, y_offset), COLOR_BORDER, 1)
        y_offset += 15

        cv2.putText(panel, "Knowledge Base:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HEADER, 1, cv2.LINE_AA)
        y_offset += 15

        # Show top 3 categories
        sorted_cats = sorted(stats['docs_per_category'].items(),
                            key=lambda x: x[1], reverse=True)[:3]

        for cat, count in sorted_cats:
            if y_offset > height - 15:
                break
            cat_text = f"  {cat}: {count}"
            if len(cat_text) > 30:
                cat_text = cat_text[:27] + "..."
            cv2.putText(panel, cat_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
            y_offset += 14

    return panel


def create_rag_compact_info(agent):
    """
    Create a compact single-line RAG status string for embedding in main visualization.

    Args:
        agent: The agent instance with RAG manager or rag_best_detection

    Returns:
        String with RAG status
    """
    has_rag_manager = hasattr(agent, 'rag_manager') and agent.rag_manager is not None
    has_rag_detection = hasattr(agent, 'rag_best_detection')

    if not has_rag_manager and not has_rag_detection:
        return "RAG: Disabled"

    # Get total docs if manager exists
    total_docs = 0
    if has_rag_manager:
        stats = agent.rag_manager.get_statistics()
        total_docs = stats['total_documents']

    # Show detection info
    if has_rag_detection and agent.rag_best_detection is not None:
        det = agent.rag_best_detection
        if has_rag_manager:
            status = f"RAG: {total_docs} docs | Best: {det['caption']} (conf={det['confidence']:.2f})"
        else:
            status = f"RAG Best: {det['caption']} (conf={det['confidence']:.2f}, dist={det['distance']:.2f}m)"
        if det['verified']:
            status += " ✓"
        else:
            status += " ✗"
    else:
        if has_rag_manager:
            status = f"RAG: {total_docs} docs | No detection"
        else:
            status = "RAG: No detection yet"

    return status
