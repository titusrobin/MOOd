import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
import matplotlib.pyplot as plt


def main():
    # Initialize dataset
    data_path = "C:/capstone_project/v1.0-trainval/"
    version = "v1.0-trainval"
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

    # Get first scene and sample
    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])

    # Get front camera data token
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data = nusc.get("sample_data", cam_token)

    # Print the camera data filename for debugging
    print("Camera data filename:", cam_data["filename"])

    # Render the sample data with annotations
    # path = nusc.get_sample_data_path(cam_token)
    # print(path)
    boxes = nusc.get_boxes(cam_token)

    # Render the sample data with boxes
    nusc.render_sample_data(
        cam_token, with_anns=True
    )  # Set with_anns=True to render annotations

    # Optionally display box labels on the image
    for box in boxes:
        print(
            f"Box Label: {box.name}, Token: {box.token}, Translation: {box.center}, Size: {box.wlh}"
        )
        # Add code to display the label on the image if required
    # nusc.render_sample_data(cam_token)


# nusc.render_annotation(cam_token)


if __name__ == "__main__":
    main()
