"""
traffic_cone_detector.py

This script defines the TrafficConeDetector class, which is responsible for training, validating, and using a YOLO object detection model to detect traffic cones in images and videos.
"""

from ultralytics import YOLO
import torch
import os
import cv2
from PIL import Image
import numpy as np
import seaborn as sns
import seaborn.objects as so
import warnings
import matplotlib.pyplot as plt


class TrafficConeDetector:
    def __init__(self, dataset_yaml, weights_path=None):
        """
        Initialize the traffic cone detector.

        Args:
            dataset_yaml (str): Path to dataset.yaml file.
            weights_path (str, optional): Path to pretrained weights.
        """
        self.dataset_yaml = dataset_yaml
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        if weights_path and os.path.exists(weights_path):
            self.model = YOLO(weights_path)
            print(f"Loaded weights from {weights_path}")
        else:
            self.model = YOLO("yolov8n.pt")  # Start with pretrained YOLOv8n model
            print("Starting with pretrained YOLOv8n model")

    def train(self, epochs=100, imgsz=640, batch_size=16):
        """
        Train the model on the custom dataset.

        Args:
            epochs (int, optional): Number of training epochs.
            imgsz (int, optional): Image size for training.
            batch_size (int, optional): Batch size for training.

        Returns:
            dict: Training results.
        """
        args = dict(
            data=self.dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=20,  # Early stopping patience
            save=True,  # Save checkpoints
            device=self.device,
            plots=True,  # Save training plots
            verbose=True,
        )

        # Start training
        results = self.model.train(**args)
        print("Training completed!")
        return results

    def validate(self):
        """
        Validate the model on the validation set.

        Returns:
            dict: Validation results.
        """
        print("Running validation...")
        results = self.model.val(data=self.dataset_yaml)
        return results

    def predict(self, image_path, conf_threshold=0.25):
        """
        Detect traffic cones in a single image.

        Args:
            image_path (str): Path to the image.
            conf_threshold (float, optional): Confidence threshold for detections.

        Returns:
            Any: Annotated image and detections.
        """
        results = self.model.predict(source=image_path, conf=conf_threshold, save=False)
        return results[0]

    def process_video(self, video_path, output_path, conf_threshold=0.25):
        """
        Process a video file and detect traffic cones.

        Args:
            video_path (str): Path to the input video.
            output_path (str): Path to save the output video.
            conf_threshold (float, optional): Confidence threshold for detections.
        """
        # Code for processing a video


def visualize_dataset(dataset_path, num_samples=5):
    """
    Visualize random samples from the dataset with their annotations.

    Args:
        dataset_path (str): Path to the dataset directory.
        num_samples (int, optional): Number of samples to visualize.
    """
 cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = self.model.predict(source=frame, conf=conf_threshold, save=False)

            # Get annotated frame
            annotated_frame = results[0].plot()

            # Write frame
            out.write(annotated_frame)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")

        cap.release()
        out.release()
        print(f"Video processing completed. Saved to {output_path}")


def visualize_dataset(dataset_path, num_samples=5):
    """
    Visualize random samples from the dataset with their annotations
    Args:
        dataset_path: Path to dataset directory
        num_samples: Number of samples to visualize
    """
    train_img_dir = os.path.join(dataset_path, "train", "images")
    train_label_dir = os.path.join(dataset_path, "train", "labels")

    # Get random samples
    img_files = os.listdir(train_img_dir)
    sample_files = np.random.choice(
        img_files, min(num_samples, len(img_files)), replace=False
    )

    for img_file in sample_files:
        # Load image
        img_path = os.path.join(train_img_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        label_path = os.path.join(train_label_dir, img_file.replace(".jpg", ".txt"))
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                labels = f.readlines()

            # Draw boxes
            h, w = img.shape[:2]
            for label in labels:
                class_id, x_center, y_center, width, height = map(
                    float, label.strip().split()
                )

                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display image
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    dataset_yaml = "cone_dataset/dataset.yaml"

    print("Visualizing dataset samples...")
    visualize_dataset("cone_dataset", num_samples=3)

    # Initialize detector
    detector = TrafficConeDetector(dataset_yaml)

    # Train the model
    results = detector.train(epochs=5)

    # Validate the model
    val_results = detector.validate()

    # Predict on a test image
    test_image = "cone_dataset/val/images/000000.jpg"
    if os.path.exists(test_image):
        results = detector.predict(test_image)

        # Display results
        plt.figure(figsize=(12, 12))
        plt.imshow(results.plot())
        plt.axis("off")
        plt.show()
