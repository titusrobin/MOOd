import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
import json
from tqdm import tqdm


class NuScenesConesDataset:
    def __init__(self, dataroot, version="v1.0-mini", split="train"):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.split = split
        self.scenes = create_splits_scenes()[split]

    def prepare_dataset(self, output_dir):
        """
        Prepare dataset in YOLO format:
        - images/train/
        - images/val/
        - labels/train/
        - labels/val/
        """
        os.makedirs(os.path.join(output_dir, "images", self.split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", self.split), exist_ok=True)

        # Track all annotations
        for scene in tqdm(self.scenes):
            scene_rec = self.nusc.get("scene", scene)
            sample = self.nusc.get("sample", scene_rec["first_sample_token"])

            while sample:
                # Get camera images
                cam_front_data = self.nusc.get(
                    "sample_data", sample["data"]["CAM_FRONT"]
                )

                # Get image
                img_path = os.path.join(self.nusc.dataroot, cam_front_data["filename"])
                img = Image.open(img_path)

                # Get annotations
                annotations = []
                for ann_token in sample["anns"]:
                    ann_rec = self.nusc.get("sample_annotation", ann_token)
                    if ann_rec["category_name"] == "movable_object.traffic_cone":
                        # Get 2D bbox in image coordinates
                        bbox = self.nusc.get_box(ann_rec["token"])
                        corners_2d = self.nusc.box_to_keypoints(
                            cam_front_data["token"], bbox
                        )

                        # Convert to YOLO format (normalized coordinates)
                        x_min, y_min = np.min(corners_2d, axis=0)
                        x_max, y_max = np.max(corners_2d, axis=0)

                        width = img.width
                        height = img.height

                        # YOLO format: <class> <x_center> <y_center> <width> <height>
                        x_center = ((x_min + x_max) / 2) / width
                        y_center = ((y_min + y_max) / 2) / height
                        bbox_width = (x_max - x_min) / width
                        bbox_height = (y_max - y_min) / height

                        annotations.append(
                            f"0 {x_center} {y_center} {bbox_width} {bbox_height}"
                        )

                # Save image and labels
                img_filename = os.path.basename(cam_front_data["filename"])
                label_filename = os.path.splitext(img_filename)[0] + ".txt"

                img.save(os.path.join(output_dir, "images", self.split, img_filename))
                with open(
                    os.path.join(output_dir, "labels", self.split, label_filename), "w"
                ) as f:
                    f.write("\n".join(annotations))

                # Move to next sample
                if sample["next"] == "":
                    break
                sample = self.nusc.get("sample", sample["next"])
