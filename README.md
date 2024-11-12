# 🐄 M🐮🐮d 🐄
MOOD: Multi-angle AI for Obscured Object Detection

Our project aims to study how different models perform in detecting objects that are partially obscured in natural settings. We have chosen traffic cones from the nuScenes dataset as our target object. NuScenes is an autonomous navigation dataset that includes multiple object classes. We selected traffic cones as the targeted object for their symmetrical shape, immobility, and small size, making them ideal for detection in complex scenes.

One specific question we aim to investigate based on this visualization is how we can design an experiment to test the effect of traffic cone visibility levels on model performance. To enhance our understanding of data distribution and design the experiment, we analyzed the visibility distribution of traffic cones. Visibility, defined as the fraction of pixels of each annotation visible across six camera feeds, is grouped into four bins: 0-40%, 40-60%, 60-80%, and 80-100%. For example, if a traffic cone is labeled as 80-100% visibility, it means the cone is almost fully or completely visible.

![Sample_1](https://github.com/user-attachments/assets/0a02fd25-5d26-4d2b-8265-1c1cb002f588)

Based on the visualization, our analysis of 1,378 traffic cones from the nuScenes mini dataset shows that 73.95% fall within the 80-100% visibility bin, indicating that most cones are highly visible. In contrast, 18.72% fall within the 0-40% visibility bin, 2.18% in the 40-60% bin, and 5.15% in the 60-80% bin. These findings suggest that our dataset is imbalanced, with a significant concentration of cones in the highest visibility bin. If we want to study the effect of traffic cone visibility levels on model performance, to ensure a balanced design for future experiments, we may need to subset the dataset to match the number of cones in the smallest visibility bin (40-60%).
