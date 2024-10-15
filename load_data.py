from nuscenes.nuscenes import NuScenes

# Specify the path to the dataset and the version you want to load
data_path = r"C:\capstone_project\v1.0-trainval01_blobs_camera"  # Replace with the path to your dataset
version = "v1.0-trainval"  # Change this if you're using a different version

# Load the nuScenes dataset
nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

# Display the first few annotated boxes for an example scene
scene = nusc.scene[0]  # Load the first scene
first_sample_token = scene["first_sample_token"]
sample = nusc.get("sample", first_sample_token)

# Print annotated boxes for the first data sample
for annotation_token in sample["anns"]:
    annotation = nusc.get("sample_annotation", annotation_token)
    print(
        f"Instance token: {annotation['instance_token']}, Category: {annotation['category_name']}"
    )
