import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import json


def load_data(
    data_path,
    version,
):
    data_path = data_path
    version = version
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

    return nusc


def get_one_scene(nusc, scene_number=0, camera_view="CAM_FRONT"):
    # Get scene and  first sample
    scene = nusc.scene[scene_number]
    sample = nusc.get("sample", scene["first_sample_token"])

    # Get front camera data token
    cam_token = sample["data"][camera_view]
    cam_data = nusc.get("sample_data", cam_token)
    print("Camera data filename:", cam_data["filename"])
    boxes = nusc.get_boxes(cam_token)

    # Render the sample data with boxes
    nusc.render_sample_data(cam_token, with_anns=True)

    # box display
    for box in boxes:
        print(
            f"Box Label: {box.name}, Token: {box.token}, Translation: {box.center}, Size: {box.wlh}"
        )
    return nusc


import json
from pathlib import Path
from tqdm import tqdm


def filter_traffic_cone_images(
    nusc, camera_views=("CAM_FRONT",), output_file="cone_samples.json"
):
    """
    Filter samples from the NuScenes dataset that contain traffic cones and save to a JSON file.

    Args:
        nusc (NuScenes): NuScenes object.
        camera_views (tuple): Camera views to consider.
        output_file (str): Path to save the filtered results as a JSON file.

    Returns:
        dict: Dictionary where keys are scene names, and values are lists of dictionaries with sample data
              containing traffic cones.
    """
    cone_samples = {}
    total_scenes = len(nusc.scene) * len(camera_views)

    # Single progress bar tracking across all scenes and views
    with tqdm(total=total_scenes, desc="Filtering traffic cones") as pbar:
        for camera_view in camera_views:
            for scene in nusc.scene:
                scene_name = scene["name"]
                scene_token = scene["token"]

                # Get all sample tokens for this scene
                sample_tokens = nusc.field2token("sample", "scene_token", scene_token)

                if scene_name not in cone_samples:
                    cone_samples[scene_name] = []

                for sample_token in sample_tokens:
                    sample = nusc.get("sample", sample_token)
                    cam_token = sample["data"].get(camera_view)

                    if cam_token:
                        data_path, boxes, _ = nusc.get_sample_data(cam_token)
                        contains_cone = any(
                            box.name == "movable_object.trafficcone" for box in boxes
                        )

                        if contains_cone:
                            cone_samples[scene_name].append(
                                {
                                    "sample_data_token": sample_token,
                                    "cam_token": cam_token,
                                    "camera_view": camera_view,
                                }
                            )

                # Update progress bar after each scene
                pbar.update(1)

    # Save the cone_samples dictionary to a JSON file
    with open(output_file, "w") as f:
        json.dump(cone_samples, f, indent=4)

    return cone_samples


def load_cone_samples(json_file: str) -> dict:
    """Load cone_samples dictionary from a JSON file."""
    with open(json_file, "r") as file:
        return json.load(file)


def save_filtered_cone_images(
    nusc: NuScenes,
    cone_samples_path: str,
    output_dir: str,
    camera_views_wanted: list,
    log_missing_tokens: bool = False,
):
    """
    Save traffic cone images from NuScenes dataset into a structured directory based on specified camera views and scenes.

    Args:
        nusc (NuScenes): NuScenes dataset instance
        cone_samples_path (str): Path to JSON file containing cone samples
        output_dir (str): Directory to save filtered images
        camera_views_wanted (list): List of camera views to organize folders
        log_missing_tokens (bool): Whether to log missing tokens and files
    """
    # Load cone_samples from JSON
    with open(cone_samples_path, "r") as file:
        cone_samples = json.load(file)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = sum(
        len(samples)
        for samples in cone_samples.values()
        if any(sample["camera_view"] in camera_views_wanted for sample in samples)
    )

    # Process and save images
    with tqdm(total=total_samples, desc="Saving traffic cone images") as pbar:
        for scene_name, samples in cone_samples.items():
            for sample in samples:
                camera_view = sample["camera_view"]

                # Only process samples in the wanted camera views
                if camera_view not in camera_views_wanted:
                    continue

                # Create camera view and scene directory within output directory
                camera_dir = output_dir / camera_view
                scene_dir = camera_dir / scene_name
                scene_dir.mkdir(parents=True, exist_ok=True)

                cam_token = sample["cam_token"]

                try:
                    # Get the sample data from NuScenes
                    sample_data = nusc.get("sample_data", cam_token)
                    image_path = Path(nusc.dataroot) / sample_data["filename"]

                    # Verify that the image file exists in the original dataset
                    if not image_path.is_file():
                        if log_missing_tokens:
                            print(f"Image file {image_path} does not exist. Skipping.")
                        pbar.update(1)
                        continue

                    # Load and save the image
                    image = Image.open(image_path)
                    output_path = scene_dir / image_path.name
                    image.save(output_path)

                    if log_missing_tokens:
                        print(f"Successfully saved {output_path}")

                except Exception as e:
                    if log_missing_tokens:
                        print(f"Error processing cam_token {cam_token}: {e}")
                finally:
                    pbar.update(1)

    if log_missing_tokens:
        print(f"\nProcessing complete. Total images saved: {total_samples}")


def main():
    # Initialize dataset
    data_path = "C:/capstone_project/v1.0-trainval/"
    version = "v1.0-trainval"
    output_dir = "C:/capstone_project/filtered_cone_images/"
    camera_views_wanted = [
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
    ]
    nusc = load_data(data_path=data_path, version=version)
    # cone_samples = filter_traffic_cone_images(nusc, camera_views=camera_views_wanted)

    # with open("temp_list.txt", "r") as file:
    # Use eval to convert each line (string) to a tuple
    # cone_samples = [eval(line.strip()) for line in file]

    # cone_samples = [item for sublist in cone_samples for item in sublist]
    save_filtered_cone_images(
        nusc, "cone_samples.json", output_dir, camera_views_wanted=camera_views_wanted
    )


if __name__ == "__main__":
    main()
