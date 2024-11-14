import os.path as osp
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.geometry_utils import box_in_image


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

    # Get the sample data path
    data_path = nusc.get_sample_data_path(cam_token)

    # Load the image using OpenCV
    image = cv2.imread(data_path)

    # Retrieve boxes for the current sample data
    boxes = nusc.get_boxes(cam_token)

    # Iterate over boxes to draw them on the image
    for box in boxes:
        # Get the box corners in image coordinates
        corners = box.corners()

        # Project the corners onto the image
        corners_2d = corners[:2, :].T  # Only take x and y coordinates

        # Draw the bounding box
        cv2.polylines(
            image, [np.int32(corners_2d)], isClosed=True, color=(0, 255, 0), thickness=2
        )

        # Display the label next to the box
        label = f"{box.name} ({box.token})"
        # Choose a position for the label
        label_position = (
            int(corners_2d[0][0]),
            int(corners_2d[0][1] - 10),
        )  # Slightly above the box
        cv2.putText(
            image,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Display the image with bounding boxes and labels
    cv2.imshow("NuScenes Sample", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
