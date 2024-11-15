"""
create_dataset.py

This script is responsible for extracting traffic cone data from the nuScenes dataset and preparing it for use with a YOLO object detection model.

The script performs the following steps:
1. Analyzes the dataset categories to understand the distribution of annotations.
2. Filters the annotations to identify traffic cones.
3. Processes the image data and annotations to generate a dataset in YOLO format.
4. Splits the dataset into training and validation sets.
5. Saves the dataset in the required directory structure.
6. Generates a dataset configuration file (dataset.yaml) for use with the YOLO model.
"""

from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import yaml
import cv2
import shutil
from pyquaternion import Quaternion


def print_category_stats(nusc):
    """
    Print statistics about annotation categories in the dataset.
    """
    categories = {}
    for sample in nusc.sample:
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            category = ann["category_name"]
            categories[category] = categories.get(category, 0) + 1

    print("\nCategory statistics:")
    for category, count in sorted(categories.items()):
        print(f"{category}: {count}")


def get_2d_bbox(nusc, sample_annotation, sample_data):
    """
    Get the 2D bounding box from a 3D annotation.
    Returns None if the box is not visible in the image.
    """
    box = Box(
        sample_annotation["translation"],
        sample_annotation["size"],
        Quaternion(sample_annotation["rotation"]),
        name=sample_annotation["category_name"],
    )

    # Get camera calibration data
    cam_data = nusc.get("sample_data", sample_data["token"])
    sensor = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    intrinsic = np.array(sensor["camera_intrinsic"])

    # Transform 3D box to sensor coordinates
    box.translate(-np.array(sensor["translation"]))
    box.rotate(Quaternion(sensor["rotation"]).inverse)

    # Project 3D box to 2D
    corners_3d = box.corners()
    corners_2d = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    # Check if box is behind camera
    if np.any(corners_3d[2, :] < 0):
        return None

    # Get 2D bounding box
    x_min, y_min = np.min(corners_2d, axis=1)
    x_max, y_max = np.max(corners_2d, axis=1)

    # Get image dimensions
    img_path = os.path.join(nusc.dataroot, cam_data["filename"])
    img = Image.open(img_path)
    img_width, img_height = img.size

    # Check if box is within image bounds
    if x_min < 0 or x_max >= img_width or y_min < 0 or y_max >= img_height:
        return None

    # Convert to YOLO format
    x_center = (x_min + x_max) / (2 * img_width)
    y_center = (y_min + y_max) / (2 * img_height)
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return [x_center, y_center, width, height]


def create_yolo_dataset(nusc, split_ratio=0.8, output_dir="cone_dataset"):
    """
    Extract traffic cone data from nuScenes and prepare it for YOLO format.
    """
    print("Starting dataset creation...")
    print("\nAnalyzing dataset categories...")
    print_category_stats(nusc)

    # Create directory structure
    for subset in ["train", "val"]:
        for subdir in ["images", "labels"]:
            path = os.path.join(output_dir, subset, subdir)
            os.makedirs(path, exist_ok=True)

    total_scenes = len(nusc.scene)
    total_images_with_cones = 0
    processed_images = 0
    cone_data = []

    print("\nProcessing scenes...")

    for scene_idx, scene in enumerate(nusc.scene):
        print(f"\nProcessing scene {scene_idx + 1}/{total_scenes}")
        sample_token = scene["first_sample_token"]

        while sample_token:
            sample = nusc.get("sample", sample_token)
            cam_front_token = sample["data"]["CAM_FRONT"]
            cam_front_data = nusc.get("sample_data", cam_front_token)

            # Get all annotations for this sample
            annotations = []
            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                # Debug print for first few annotations
                if processed_images < 1:
                    print(f"Debug - Annotation category: {ann['category_name']}")
                annotations.append(ann)

            # Filter for traffic cones (check for both possible category names)
            cone_annotations = [
                ann
                for ann in annotations
                if (
                    "traffic.cone" in ann["category_name"]
                    or "traffic_cone" in ann["category_name"]
                    or "cone" in ann["category_name"].lower()
                )
            ]

            if cone_annotations:
                print(f"Found {len(cone_annotations)} cones in image")
                img_path = os.path.join(nusc.dataroot, cam_front_data["filename"])
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found at {img_path}")
                    continue

                valid_boxes = []
                for ann in cone_annotations:
                    bbox = get_2d_bbox(nusc, ann, cam_front_data)
                    if bbox is not None:
                        valid_boxes.append(bbox)

                if valid_boxes:
                    cone_data.append({"image_path": img_path, "boxes": valid_boxes})
                    total_images_with_cones += 1

            processed_images += 1
            if processed_images % 100 == 0:
                print(
                    f"Processed {processed_images} images, found {total_images_with_cones} with cones"
                )

            sample_token = sample["next"]

    print(f"\nTotal images processed: {processed_images}")
    print(f"Total images with cones: {total_images_with_cones}")

    if not cone_data:
        print("No traffic cones found in the dataset!")
        return

    # Split and save data
    np.random.shuffle(cone_data)
    split_idx = int(len(cone_data) * split_ratio)
    train_data = cone_data[:split_idx]
    val_data = cone_data[split_idx:]

    print(f"\nSplitting dataset: {len(train_data)} train, {len(val_data)} val")

    for subset, data in [("train", train_data), ("val", val_data)]:
        print(f"\nProcessing {subset} set...")
        for idx, item in enumerate(data):
            src_path = item["image_path"]
            dst_path = os.path.join(output_dir, subset, "images", f"{idx:06d}.jpg")
            shutil.copy2(src_path, dst_path)

            label_path = os.path.join(output_dir, subset, "labels", f"{idx:06d}.txt")
            with open(label_path, "w") as f:
                for box in item["boxes"]:
                    f.write(f"0 {' '.join(map(str, box))}\n")

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    dataset_config = {
        "path": os.path.abspath(output_dir),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "traffic_cone"},
    }

    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, sort_keys=False)

    print(f"\nDataset creation completed!")


if __name__ == "__main__":
    dataroot = "../data/v1.0-mini"  # Replace with your NuScenes data path
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)
    create_yolo_dataset(nusc)
