# MOOD: Multi-View Obscured Object Detection
#### Duke University MIDS Capstone Project 2024-2025
**Authors:** Robin Arun, Katie Hucker, Afraa Noureen, Jiayi Zhou

## I. Project Overview 

This project will develop an object detection tool to identify partially obscured objects while leveraging multiple images or views of the object. Current single-view detection methods lack in the ability to render scene contextualization, or reference images near by into its object detection capabilities. We believe by using multi-view modeling techniques obscured objects can be identified quicker, more accurately, and with less dense data. Multi-view models are more computationally complex, therefore, we wish to understand the relationship and improvement when compared to single view detection methods. 

To address this, we will first establish a baseline model, single-view model. We can then advance tomulti-view models like DETR3D, which integrates multiple perspectives to the object detection method. Our approach includes: filtering labeled to obscured objects of interest, implementing the baseline and multi-view models, analysis of methods and performance answering stakeholder driven questions.

You can find further discussion within our write-up document linked below.

Here is the file structure of our software package: 

> ```
> - Baseline_YOLO/
>   ├── code/
>   │   ├── YOLO_Documentation.md
>   │   ├── dataset_creation.py
>   │   └── traffic_cone_detector.py
> 
> - Data_Toolkit/
>   ├── nuscenes/
>   │   ├── cone_samples.json
>   │   └── load_data.py
>   └── toolkit_instructions.txt
> 
> - In_Progress_DETR3D/
>   ├── 00_setup_downloads.sh
>   ├── 10_setup_installs.sh
>   ├── Afraa_DETR3D_Trial.ipynb
>   ├── PETR.ipynb
>   └── detr_base_w_nuScenes.ipynb
> 
> - communications/
>   └── 11_8_capstone_class.pdf
> ```

## II. Semester Contributions and Findings

This section describes what is within this repository and our deliverables for the Fall 2024 Semester. 

1. Write Up: INSERT LINK HERE
2. Dataset Prep Toolkit: Dataset filtering and saving
3. Data Discussion and Results
4. YOLO Baseline Model Implementation: Takes dataset, prepares for YOLO model, trains and tests data


### 1. Write Up
INSERT LINK HERE

### 2. Dataset Prep Toolkit 

You will find the toolkit folder within the repository. The following steps is what the toolkit does:

- Loads the nuScenes Dataset using the nuScenes devkit
- Filters the dataset to only look for traffic cones
- Saves the images which only contains traffic cones in a new file structure

Time note: 85 scenes take approximately 45 mins to filter. 

### 3. Data Discussion and Results

To evaluate how model performance varies across different levels of occlusion, we analyzed the visibility distribution of traffic cones within the nuScenes dataset. Due to the large size of the full dataset and the computational limitations we face, we chose to conduct our experiments using the mini dataset, which is a smaller but representative subset of the full nuScenes collection. This allowed us to perform meaningful analyses within our resource constraints while maintaining relevance to the overall dataset’s characteristics. The mini dataset mirrors the visibility distribution found in the complete nuScenes dataset, making it a practical choice for gaining insights into model performance.  

Our analysis of the 1,378 traffic cones in the mini dataset revealed the following visibility distribution:  

![Final_Updated](https://github.com/user-attachments/assets/502364c6-8e96-40bc-971f-0596e2a0905b)

The key findings from our analysis include:  
- **Visibility Distribution**: Our analysis of the 1,378 traffic cones in the nuScenes mini dataset revealed a significant imbalance in visibility. Most cones (74%) were fully or mostly visible (80-100% visibility), while a much smaller proportion (2%) were partially occluded (40-60% visibility). The remainder of the cones fell into the 0-40% or 60-80% visibility bins, with 19% and 5%, respectively.  
- **Challenges with Occlusion**: The dataset shows that while most cones are highly visible, partial occlusion is less frequently represented. This suggests that while models may perform well when cones are fully visible, they may struggle in real-world environments where occlusions are more common.  
- **Data Imbalance**: The underrepresentation of partially occluded cones (especially in the 40-60% visibility range) presents a challenge for training robust models. We are exploring methods to balance the dataset to ensure more realistic evaluation of model performance across different visibility conditions.  

This analysis highlights the importance of balanced datasets to ensure reliable detection in real-world scenarios, particularly when occlusion is present. For more detailed information on the dataset composition and methodology, refer to the **Data Overview** section of the write-up.

### 4. YOLO Baseline Model Implementation
You will find the YOLO Baseline Model folder within the repository called `Baseline_YOLO/code`. YOLO object detection model is used as baseline to detect traffic cones. It includes scripts to extract and prepare a dataset from the nuScenes dataset, as well as to train, validate, and use the detection model.

#### Files

1. `create_dataset.py`: This script is responsible for extracting traffic cone data from the nuScenes dataset and preparing it for use with a YOLO object detection model.
2. `traffic_cone_detector.py`: This script defines the `TrafficConeDetector` class, which is used for training, validating, and using the YOLO object detection model to detect traffic cones.
3. More detail introductions of each file is included called `YOLO_Documentation.md` under the folder `Baseline_YOLO/code`.

## III. How to launch the project

### 1. Prepare Data
   - Download the nuScenes dataset from the [nuScenes website](https://www.nuscenes.org/).
   - Extract the dataset and place it in the root directory (outside the `MOOd` folder) to match the expected folder structure:
     ```
     project_root/
     ├── v1.0-trainval/    # Folder containing dataset images and related files
     └── MOOd/             # Folder containing toolkit and scripts
     ```
   - Ensure that `v1.0-trainval` holds the dataset and `MOOd/nuscenes` holds the DevKit.
   - Open a terminal, navigate to the `MOOd` folder, and execute:
     ```bash
     python load_data.py
     ```
   - This will load and filter the nuScenes data according to the configurations in `load_data.py`.

For more information, please see the [Toolkit README](MOOd/toolkit_instructions.txt).

### 2. Baseline Model--YOLO
#### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- `torch`, `opencv-python`, `Pillow`, `seaborn`, `matplotlib`, and `ultralytics` packages (for YOLOv8)

Install the required packages:
```bash
pip install torch ultralytics opencv-python pillow seaborn matplotlib
```

#### Usage
1. Ensure you have the nuScenes dataset downloaded and the path set in `Baseline_YOLO/code`.
2. Run `create_dataset.py` to generate the YOLO dataset.
4. Customize the training parameters in `traffic_cone_detector.py` as needed.
5. Run `traffic_cone_detector.py` to train the model, validate it, and test it on a sample image.

#### Usage Example
```python
from create_dataset import create_yolo_dataset
from nuscenes import NuScenes
from traffic_cone_detector import TrafficConeDetector

# Set the path to your nuScenes dataset
dataroot = "../data/v1.0-mini"  

# Initialize the NuScenes instance
nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)

# Create the YOLO dataset
create_yolo_dataset(nusc)

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
## Next Steps

#### 1. Train and test YOLO model on the full filtered dataset

The YOLO model was train and tested with randomized very small sample of the original data. We finally have all the data subset to just include traffic cones. We will able to train and test using the YOLO model software already created.

#### 2. DETR3D Model Implementation
   
We will finish implementation of our multi-view model, DETR3D. We expect to have this done by the end of the semester, but we were unable to complete within the 11/15 deadline. We are almost there and will have a similar software toolkit to the YOLO model which is already provided. 

#### 3. Train and test DETR3D model on full filtered dataset

After the modeling code is compelete and works on the subset dataset, we can train and test using the full filtered dataset

#### 4. Return to our driving questions. 
Our driving questions assess the models deeply. These our questions our stakeholder hopes to understand and we will continue to explore. The questions hope to provide answers about the following topics: downsampled data, obscurement level of object vs performance, model explainability via feature analysis.
