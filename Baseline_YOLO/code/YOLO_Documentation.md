# Baseline YOLO

## Technical Documentation

### `create_dataset.py`

This script extracts traffic cone data from the nuScenes mini dataset and prepares it for use with a YOLO object detection model.

#### Key functionality:
1. Analyzes dataset categories and prints annotation statistics.
2. Filters annotations to identify traffic cones.
3. Processes image data and annotations to generate a YOLO-formatted dataset.
4. Splits the dataset into training and validation sets.
5. Saves the dataset in the required directory structure.
6. Generates a dataset configuration file (`dataset.yaml`) for use with the YOLO model.

#### To use this script:
1. Set the `dataroot` variable to the path of your nuScenes dataset.
2. Run the script to create the `cone_dataset` directory with training and validation data.

```python
from nuscenes import NuScenes
from create_dataset import create_yolo_dataset

# Set the path to your nuScenes dataset
dataroot = "../data/v1.0-mini"  

# Initialize the NuScenes instance
nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)

# Create the YOLO dataset
create_yolo_dataset(nusc)
```

### `traffic_cone_detector.py`

This script defines the `TrafficConeDetector` class, which is used for training, validating, and using a YOLO object detection model to detect traffic cones.

#### Key functionality:
1. Initializes the detector with the dataset configuration and optional pre-trained weights.
2. Provides a `train()` method to train the model on the custom dataset.
3. Provides a `validate()` method to evaluate the model on the validation set.
4. Provides a `predict()` method to detect traffic cones in a single image.
5. Provides a `process_video()` method to detect traffic cones in a video file.
6. Includes a `visualize_dataset()` function to display random samples from the dataset with their annotations.

#### To use this script:
1. Ensure the `cone_dataset/dataset.yaml` file is available.
2. Customize the training parameters as needed.
3. Run the script to train the model, validate it, and test it on a sample image or video.

## Usage Example

```python
from traffic_cone_detector import TrafficConeDetector

# Initialize detector
detector = TrafficConeDetector("cone_dataset/dataset.yaml")

# Train the model
detector.train(epochs=5)

# Validate the model
val_results = detector.validate()

# Predict on a test image
test_image = "cone_dataset/val/images/000000.jpg"
results = detector.predict(test_image)

# Display the results
plt.figure(figsize=(12, 12))
plt.imshow(results.plot())
plt.axis("off")
plt.show()
```

### Troubleshooting

- **GPU Not Detected**: Ensure you have a compatible NVIDIA GPU and CUDA installed.
- **Data Not Found**: Confirm that the nuScenes dataset is downloaded and `create_dataset.py` ran successfully.

### Notes

- Adjust parameters like `epochs`, `batch_size`, and `conf_threshold` in `traffic_cone_detector.py` to tune performance.
- You may need to modify the paths to match your file system structure.
