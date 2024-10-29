from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, Quaternion
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Specify the path to the dataset and the version you want to load
data_path = r"C:\capstone_project\v1.0-trainval_blobs_camera"
version = "v1.0-trainval"

# Load the nuScenes dataset
nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

# Load the first scene and sample
scene = nusc.scene[0]
first_sample_token = scene["first_sample_token"]
sample = nusc.get("sample", first_sample_token)

# Load the image for the first sample
image_token = sample["data"]["CAM_FRONT"]
image = nusc.get("sample_data", image_token)
image_path = os.path.join(
    data_path, "v1.0-trainval", image["filename"].replace("/", "\\")
)

# Check if the image file exists before opening
if os.path.exists(image_path):
    img = Image.open(image_path)
else:
    print(f"File not found: {image_path}")

# Plot the image
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img)

# Camera intrinsic matrix as a NumPy array
cam_intrinsic = np.array(
    nusc.get("calibrated_sensor", image["calibrated_sensor_token"])["camera_intrinsic"]
)

# Set to track drawn annotation tokens to avoid duplicates
drawn_tokens = set()

# Iterate over all annotations in the sample
for annotation_token in sample["anns"]:
    annotation = nusc.get("sample_annotation", annotation_token)
    category_name = annotation["category_name"]
    print(category_name)

    # Extract translation, size, and rotation
    translation, size, rotation = (
        annotation["translation"],
        annotation["size"],
        annotation["rotation"],
    )

    print(
        f"Token: {annotation_token}, Translation: {translation}, Size: {size}, Rotation: {rotation}"
    )

    # Create a Box object with the annotation information
    box = Box(translation, size, Quaternion(rotation))

    # Project the box corners onto the 2D image plane
    corners_2d = view_points(box.corners(), cam_intrinsic, normalize=True)
    print(f"Corners 2D: {corners_2d}")

    # Ensure the corners are within image dimensions
    if (
        np.any(corners_2d[0, :] < 0)
        or np.any(corners_2d[0, :] > img.width)
        or np.any(corners_2d[1, :] < 0)
        or np.any(corners_2d[1, :] > img.height)
    ):
        print(f"Box {annotation_token} is out of image bounds.")

    # Draw bounding box if visible
    if BoxVisibility.ANY:  # Update this as necessary for your visibility requirements
        if annotation_token not in drawn_tokens:
            # Calculate the min and max corners for a rectangular bounding box
            x_min, y_min = corners_2d[0, :].min(), corners_2d[1, :].min()
            x_max, y_max = corners_2d[0, :].max(), corners_2d[1, :].max()

            # Draw the bounding box as a rectangle
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            drawn_tokens.add(annotation_token)  # Mark this annotation as drawn

            # Add a label for the object
            ax.text(
                x_min,
                y_min,
                category_name,
                color="white",
                fontsize=10,
                bbox=dict(facecolor="red", edgecolor="none", pad=1),
            )

plt.axis("off")
plt.show()
