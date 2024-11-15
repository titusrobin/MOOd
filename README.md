# MOOD: Multi-View Obscured Object Detection
#### Duke University MIDS Capstone Project 2024-2025
**Authors:** Robin Arun, Katie Hucker, Afraa Noureen, Jiayi Zhou

## Project Overview 

This project will develop an object detection tool to identify partially obscured objects while leveraging multiple images or views of the object. Current single-view detection methods lack in the ability to render scene contextualization, or reference images near by into its object detection capabilities. We believe by using multi-view modeling techniques obscured objects can be identified quicker, more accurately, and with less dense data. Multi-view models are more computationally complex, therefore, we wish to understand the relationship and improvement when compared to single view detection methods. 

To address this, we will first establish a baseline model, single-view model. We can then advance tomulti-view models like DETR3D, which integrates multiple perspectives to the object detection method. Our approach includes: filtering labeled to obscured objects of interest, implementing the baseline and multi-view models, analysis of methods and performance answering stakeholder driven questions.

### Motivation 
Hidden objects pose significant risks in critical environments, where not finding a partially obscured object can have high-stakes consequences. Some of these high stakes scenarios include: autonoumous car navigation, landmine detection, and search and rescue missions. Traditional object detection methods struggle in these scenarios, often requiring multiple data collects, with the possibility of still not finding the objects due to obscurement. Current multi-view methods are computationally heavy requiring 3D point clouds or rendered scenes. Therefore, this project introduces a '3D interpretation of scenes' to enhance detection accuracy without the heavy overhead of full 3D reconstruction, providing an efficient solution for real-world environments.

### Goal
Develop a model that can accurately detect and classify partially obscured objects by integrating multiple images of a scene, leveraging the spatial relationships among these views for robust detection.

Driving Questions:
- To what extent can multi-view data improve detection accuracy for partially obscured objects?
- How does varying the level of object obscurity impact detection accuracy in multi-view models?
- What is the minimum number of scene views required to achieve reliable detection?
- Which object features most significantly contribute to accurate detection?

## Semester Contributions and Findings

This section describes what is within this repository and our deliverables for the Fall 2024 Semester. 

1. Write Up: INSERT LINK HERE
3. Dataset Prep Toolkit: Dataset filtering and saving
4. YOLO Baseline Model Implementation: Takes dataset, prepares for YOLO model, trains and tests data


### 1. Write Up
INSERT LINK HERE

### 2. Data Introduction and Exploration

Our project aims to evaluate the performance of different models in detecting objects that are partially obscured in natural settings, with a specific focus on traffic cones from the nuScenes dataset. NuScenes is a comprehensive autonomous driving research dataset. It’s got 1,000 20-second urban driving scenes collected in Boston and Singapore, featuring synchronized sensor data from 6 cameras, 1 LiDAR, 5 RADAR sensors, GPS, and IMU (Inertial Measurement Unit - helps with camera movement and orientation data), which helps capture orientation data for accurate scene understanding. They provide all cameras, LiDAR, RADAR and camera intrinsic and extrinsic data across dozens of objects per scene. The camera data expands over 1.4 million images at 2Hz (twice per second), including annotations from 23 object classes (incl. cars, pedestrians, bicycles) that were annotated by humans. NuScenes is a benchmark dataset for many pioneering research models on multi-view object detection, presenting experimentation opportunities like assessing whether the number of images improves detection accuracy, amongst other options as mentioned in the proposal above. 

Among 23 object classes, we selected traffic cones due to their symmetrical shape, immobility, and small size, which present unique challenges in complex scenes. To better understand our data and design effective experiments, we analyzed the visibility distribution of traffic cones—measured as the fraction of visible pixels across six camera feeds. In the nuScenes dataset, visibility levels are grouped into four bins: 0-40%, 40-60%, 60-80%, and 80-100%. For example, a cone with 80-100% visibility is nearly or fully visible. One specific question we aim to investigate based on this visualization is how we can design an experiment to test the effect of traffic cone visibility levels on model performance.

![Final_Updated](https://github.com/user-attachments/assets/502364c6-8e96-40bc-971f-0596e2a0905b)

Specifically, we examined 1,378 traffic cones from the nuScenes mini dataset, finding that 74% fall in the 80-100% visibility bin, indicating most cones are highly visible. In contrast, 19% are in the 0-40% bin, 2% in the 40-60% bin, and 5% in the 60-80% bin. These findings reveal an imbalance in visibility levels, with a concentration of cones in the highest visibility bin. The predominance of highly visible cones (74%) enables our model to learn full representations of the data, supporting broader research goals. However, to ensure a balanced approach in studying visibility effects on model performance, we may need to subset the dataset to align with the number of cones in the smallest visibility bin (40-60%).

### 2. Dataset Prep Toolkit 

You will find the toolkit folder within the repository. The following steps is what the toolkit does:

- Loads the nuScenes Dataset using the nuScenes devkit
- Filters the dataset to only look for traffic cones
- Saves the images which only contains traffic cones in a new file structure

Time note: 85 scenes take approximately 45 mins to filter. 

### 3. YOLO Baseline Model Implementation
You will find the YOLO Baseline Model folder within the repository called `Baseline_YOLO/code`. YOLO object detection model is used as baseline to detect traffic cones. It includes scripts to extract and prepare a dataset from the nuScenes dataset, as well as to train, validate, and use the detection model.

#### Files

1. `create_dataset.py`: This script is responsible for extracting traffic cone data from the nuScenes dataset and preparing it for use with a YOLO object detection model.
2. `traffic_cone_detector.py`: This script defines the `TrafficConeDetector` class, which is used for training, validating, and using the YOLO object detection model to detect traffic cones.

## How to launch the project

### Prepare Data

### Baseline Model--YOLO
#### Usage
1. Ensure you have the nuScenes dataset downloaded and the path set in `Baseline_YOLO/code/create_dataset.py`.
2. Run `Baseline_YOLO/code/create_dataset.py` to generate the YOLO dataset.
3. Customize the training parameters in `Baseline_YOLO/code/traffic_cone_detector.py` as needed.
4. Run `Baseline_YOLO/code/traffic_cone_detector.py` to train the model, validate it, and test it on a sample image or video.

## Next Steps

#### 1. Train and test YOLO model on the full filtered dataset

The YOLO model was train and tested with randomized very small sample of the original data. We finally have all the data subset to just include traffic cones. We will able to train and test using the YOLO model software already created.

#### 2. DETR3D Model Implementation
   
We will finish implementation of our multi-view model, DETR3D. We expect to have this done by the end of the semester, but we were unable to complete within the 11/15 deadline. We are almost there and will have a similar software toolkit to the YOLO model which is already provided. 

#### 3. Train and test DETR3D model on full filtered dataset

After the modeling code is compelete and works on the subset dataset, we can train and test using the full filtered dataset

#### 4. Return to our driving questions. 
Our driving questions assess the models deeply. These our questions our stakeholder hopes to understand and we will continue to explore. The questions hope to provide answers about the following topics: downsampled data, obscurement level of object vs performance, model explainability via feature analysis.
