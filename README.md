# Personal-Protective-Equipment-Detection-using-YOLOv10
Modeling the impact of dataset size and class imbalance on YOLOv10-based PPE detection systems.

‎Title – Modeling the impact of dataset size and class imbalance on YOLOv10-based PPE detection systems.

‎Description – This paper investigates the influence of dataset size and class imbalance on the performance of YOLOv10 based personal protective equipment (PPE) detection systems. Six YOLOv10 configurations (n, s, m, b, l, x) were tested on domain-specific datasets covering construction, industrial, and medical contexts. A novel mathematical model is introduced to describe the nonlinear relationship between dataset size and detection performance (mAP@50), revealing a saturation threshold beyond which additional data yield diminishing returns (R² = 0.968). The analysis also highlights that the superior accuracy of complex YOLOv10-x models significantly declines under conditions of pronounced class imbalance. These findings underscore the importance of balanced and sufficiently large datasets in optimizing detection accuracy for real-world safety applications.
 
‎Dataset Information: [CHV Dataset](https://github.com/ZijianWang1995/ppe_detection)
‎
Code Information: In this repository you can find the yolo refine block experiment results.
‎
Usage Instructions – Add yolo refine block arch. and run YOLO models.

‎Requirements – python >=3.8 and ultralytics>=8.1.0, PyTorch>=1.8.
