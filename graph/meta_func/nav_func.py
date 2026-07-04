import numpy as np
from collections import Counter
from PIL import Image
import math
import torch
from configs.detector_config import DETECTOR_TYPE


class NavFunc():
    def __init__(self):
        pass

    def find_modes(self, lst):
        if len(lst) == 0:
            return ['object']
        else:
            counts = Counter(lst)
            max_count = max(counts.values())
            modes = [item for item, count in counts.items() if count == max_count]
            return modes

    def get_joint_image(self, node1, node2):
        image_idx1 = node1.object["image_idx"]
        image_idx2 = node2.object["image_idx"]
        image_idx = set(image_idx1) & set(image_idx2)
        if len(image_idx) == 0:
            return None
        conf_max = -np.inf
        # get joint images of the two nodes
        for idx in image_idx:
            conf1 = node1.object["conf"][image_idx1.index(idx)]
            conf2 = node2.object["conf"][image_idx2.index(idx)]
            conf = conf1 + conf2
            if conf > conf_max:
                conf_max = conf
                idx_max = idx
        image = self.segment2d_results[idx_max]["image_rgb"]
        image = Image.fromarray(image)
        return image

    def discriminate_relation(self, edge):
        image = self.get_joint_image(edge.node1, edge.node2)
        if image is not None:
            response = self.get_vlm_response(self.prompt_discriminate_relation.format(edge.node1.caption, edge.node2.caption, edge.relation), image)
            if 'yes' in response.lower():
                return True
            else:
                return False
        else:
            if edge.node1.room_node != edge.node2.room_node:
                return False
            x1, y1 = edge.node1.center
            x2, y2 = edge.node2.center
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance > self.map_size // 40:
                return False
            alpha = math.atan2(y2 - y1, x2 - x1)
            sin_2alpha = 2 * math.sin(alpha) * math.cos(alpha)
            if not -0.05 < sin_2alpha < 0.05:
                return False
            n = 3
            for i in range(1, n):
                x = int(x1 + (x2 - x1) * i / n)
                y = int(y1 + (y2 - y1) * i / n)
                if not self.free_map[y, x]:
                    return False
            return True

    def perception(self):
        if not self.agent.found_goal:
            self.agent.detect_objects(self.observations)
            if self.agent.total_steps % 2 == 0:
                # Room detection - support GLIP, FastSAM+CLIP, and FastSAM+text_prompt
                if DETECTOR_TYPE == 'glip':
                    room_detection_result = self.agent.glip_demo.inference(
                        self.observations["rgb"][:,:,[2,1,0]],
                        self.agent.rooms_captions
                    )
                elif DETECTOR_TYPE in ['fastsam_clip', 'fastsam_text']:
                    # Use FastSAM for room detection (both CLIP and text_prompt modes)
                    image_rgb = self.observations["rgb"][:,:,[2,1,0]]
                    room_categories = [r.strip() for r in self.agent.rooms_captions.rstrip('.').split('. ')]
                    room_results = self.agent.detector.detect(image_rgb, room_categories)

                    # Convert to GLIP-like format
                    class MockPredictions:
                        def __init__(self, bboxes, labels, scores):
                            self.bbox = torch.tensor(bboxes) if not isinstance(bboxes, torch.Tensor) else bboxes
                            if isinstance(labels, list):
                                class LabelWrapper:
                                    def __init__(self, label_list):
                                        self.labels = label_list
                                    def tolist(self):
                                        return self.labels
                                    def __len__(self):
                                        return len(self.labels)
                                    def __getitem__(self, idx):
                                        return self.labels[idx]
                                labels = LabelWrapper(labels)
                            self._fields = {'labels': labels, 'scores': torch.tensor(scores) if not isinstance(scores, torch.Tensor) else scores}
                        def get_field(self, name):
                            return self._fields.get(name)

                    room_detection_result = MockPredictions(
                        room_results['bboxes'],
                        room_results['labels'],
                        room_results['scores']
                    )
                else:
                    raise ValueError(f"Unknown detector type: {DETECTOR_TYPE}")

                self.agent.update_room_map(self.observations, room_detection_result)
