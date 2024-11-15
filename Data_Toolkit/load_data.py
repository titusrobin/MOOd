import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import json


def load_data(data_path: str, version: str) -> NuScenes:
    """
    Load NuScenes dataset.

    Args:
        data_path (str): Path to NuScenes dataset directory.
        version (str): Dataset version, e.g., 'v1.0-trainval'.

    Returns:
        NuScenes: Loaded NuScenes dataset instance.
    """
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    return nusc


def get_one_scene(
    nusc: NuScenes, scene_number: int = 0, camera_view: str = "CAM_FRONT"
) -> NuScenes:
    """
    Display the first sample of a specified scene and camera view with bounding boxes.

    Args:
        nusc (NuScenes): NuScenes dataset instance.
        scene_number (int, optional): Scene index to display. Defaults to 0.
        camera_view (str, optional): Camera view to use. Defaults to "CAM_FRONT".

    Returns:
        NuScenes: NuScenes instance (unmodified).
    """
    # Get specified scene and its first sample
    scene = nusc.scene[scene_number]
    sample = nusc.get("sample", scene["first_sample_token"])

    # Retrieve front camera data
    cam_token = sample["data"][camera_view]
    cam_data = nusc.get("sample_data", cam_token)
    print("Camera data filename:", cam_data["filename"])
    boxes = nusc.get_boxes(cam_token)

    # Render the sample data with annotations
    nusc.render_sample_data(cam_token, with_anns=True)

    # Display each detected box in the camera view
    for box in boxes:
        print(
            f"Box Label: {box.name}, Token: {box.token}, Translation: {box.center}, Size: {box.wlh}"
        )
    return nusc


def filter_traffic_cone_images(
    nusc: NuScenes,
    camera_views: tuple = ("CAM_FRONT",),
    output_file: str = "cone_samples.json",
) -> dict:
    """
    Filter samples that contain traffic cones and save results to a JSON file.

    Args:
        nusc (NuScenes): NuScenes dataset instance.
        camera_views (tuple): Camera views to search for traffic cones.
        output_file (str): Path to output JSON file.

    Returns:
        dict: Dictionary where keys are scene names, and values are lists of sample data containing traffic cones.
    """
    cone_samples = {}
    total_scenes = len(nusc.scene) * len(camera_views)

    with tqdm(total=total_scenes, desc="Filtering traffic cones") as pbar:
        for camera_view in camera_views:
            for scene in nusc.scene:
                scene_name = scene["name"]
                scene_token = scene["token"]

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

                pbar.update(1)

    with open(output_file, "w") as f:
        json.dump(cone_samples, f, indent=4)

    return cone_samples


def load_cone_samples(json_file: str) -> dict:
    """
    Load a JSON file containing filtered cone samples.

    Args:
        json_file (str): Path to JSON file with cone samples.

    Returns:
        dict: Loaded dictionary with cone samples data.
    """
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
    Save images containing traffic cones from specified camera views into structured directories.

    Args:
        nusc (NuScenes): NuScenes dataset instance.
        cone_samples_path (str): Path to JSON file with filtered cone samples.
        output_dir (str): Directory to save cone images.
        camera_views_wanted (list): List of camera views to filter and save.
        log_missing_tokens (bool): If True, log missing image files.
    """
    with open(cone_samples_path, "r") as file:
        cone_samples = json.load(file)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = sum(
        len(samples)
        for samples in cone_samples.values()
        if any(sample["camera_view"] in camera_views_wanted for sample in samples)
    )

    with tqdm(total=total_samples, desc="Saving traffic cone images") as pbar:
        for scene_name, samples in cone_samples.items():
            for sample in samples:
                camera_view = sample["camera_view"]

                if camera_view not in camera_views_wanted:
                    continue

                camera_dir = output_dir / camera_view
                scene_dir = camera_dir / scene_name
                scene_dir.mkdir(parents=True, exist_ok=True)

                cam_token = sample["cam_token"]

                try:
                    sample_data = nusc.get("sample_data", cam_token)
                    image_path = Path(nusc.dataroot) / sample_data["filename"]

                    if not image_path.is_file():
                        if log_missing_tokens:
                            print(f"Image file {image_path} does not exist. Skipping.")
                        pbar.update(1)
                        continue

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
    """
    Main function to initialize NuScenes dataset, filter traffic cone images, and save filtered images.

    Args:
        None (hardcoded paths are used for demonstration purposes).
    """
    # Initialize dataset parameters
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

    # Load dataset and filter traffic cone images
    nusc = load_data(data_path=data_path, version=version)

    # Uncomment if filtering is needed:
    # cone_samples = filter_traffic_cone_images(nusc, camera_views=camera_views_wanted)

    # Save filtered cone images
    save_filtered_cone_images(
        nusc, "cone_samples.json", output_dir, camera_views_wanted=camera_views_wanted
    )


if __name__ == "__main__":
    main()
