# 🐄 M🐮🐮d 🐄
MOOD: Multi-angle AI for Obscured Object Detection

<<<<<<< HEAD
### Katie's filepath just for awareness...
![alt text](image.png)
=======
Our project aims to study how different models perform in detecting objects that are partially obscured in natural settings. We have chosen traffic cones from the nuScenes dataset as our target object. NuScenes is an autonomous navigation dataset that includes multiple object classes. We selected traffic cones as the targeted object for their symmetrical shape, immobility, and small size, making them ideal for detection in complex scenes.

One specific question raised by our stakeholders is whether we can study how changes in the visibility levels of traffic cones affect model performance. To address this, we aim to design an experiment that investigates this relationship. As a first step, we analyzed the distribution of visibility levels for traffic cones. Visibility is defined as the fraction of pixels of each annotation visible across six camera feeds and is categorized into four bins: 0-40%, 40-60%, 60-80%, and 80-100%. For instance, a traffic cone with 80-100% visibility indicates that it is almost fully or completely visible. This analysis will help inform the design of our experiment and deepen our understanding of the data distribution.

![Final_Updated](https://github.com/user-attachments/assets/502364c6-8e96-40bc-971f-0596e2a0905b)

Based on the visualization, our analysis of 1,378 traffic cones from the nuScenes mini dataset shows that 74% fall within the 80-100% visibility bin, indicating that most cones are highly visible. In contrast, 19% fall within the 0-40% visibility bin, 2% in the 40-60% bin, and 5% in the 60-80% bin. These findings suggest that our dataset is imbalanced, with a significant concentration of cones in the highest visibility bin. If we want to study the effect of traffic cone visibility levels on model performance, to ensure a balanced design for future experiments, we may need to subset the dataset to match the number of cones in the smallest visibility bin (40-60%).
>>>>>>> origin/main
