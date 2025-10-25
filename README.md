# Personal-Protective-Equipment-Detection-using-YOLOv10
Modeling the impact of dataset size and class imbalance on YOLOv10-based PPE detection systems.

‎Title – Modeling the impact of dataset size and class imbalance on YOLOv10-based PPE detection systems.

‎Description – This paper investigates the influence of dataset size and class imbalance on the performance of YOLOv10 based personal protective equipment (PPE) detection systems. Six YOLOv10 configurations (n, s, m, b, l, x) were tested on domain-specific datasets covering construction, industrial, and medical contexts. A novel mathematical model is introduced to describe the nonlinear relationship between dataset size and detection performance (mAP@50), revealing a saturation threshold beyond which additional data yield diminishing returns (R² = 0.968). The analysis also highlights that the superior accuracy of complex YOLOv10-x models significantly declines under conditions of pronounced class imbalance. These findings underscore the importance of balanced and sufficiently large datasets in optimizing detection accuracy for real-world safety applications.
 
‎Dataset Information: 

* [CHV Dataset](https://github.com/ZijianWang1995/ppe_detection)
* [SHELK5K](https://universe.roboflow.com/database-sjrvw/shel5k-new)
* [SH17](https://github.com/ahmadmughees/SH17dataset)


Code Information: In this repository you can find the yolo refine block experiment results.
‎
Usage Instructions – Add yolo refine block arch. and run YOLO models.

‎Requirements – python >=3.8 and ultralytics>=8.1.0, PyTorch>=1.8.




 ### The Impact of Dataset Size and Distribution on Model Training Performance
 The graphs definitely show how mAP@50, Precision, and Recall relate to dataset size. This is because
 they assess the influence of dataset size on training results (see Figure 4). Training the model on different
 volumes of data allowed us to examine how the rising image count influences YOLOv10’s object detection
 performance.
 The graphs presented show the results of polynomial regression for various YOLOv10 configura
tions (n, s, m, b, l). As the dataset size increases, it is clear that precision, recall and mAP@50 all
 improve. However, beyond a certain point, further increases in dataset volume yield diminishing returns
 in performance gains. This indicates that the model is reaching a saturation threshold.
 These findings make it clear: you must balance dataset size during model training. Insufficient data
 leads to underfitting and low detection quality. Excessively large datasets do not necessarily lead to
 significant improvements in accuracy and can incur higher computational costs.
 The observed nonlinear nature of this dependency, along with the influence of additional factors
 such as pronounced class imbalance and neural network configuration, necessitated the proposal of a
 mathematical model. This model was designed to quantitatively describe and interpret these relationships.
 The model definitely captures the combined effect of dataset size, class imbalance, and YOLOv10
 configuration on detection accuracy:
 M=β0+β1·V+β2·V2+β3·D+β4·C1+β5·C2+β6·C3+β7·C4+β8·C5+ε
 
 where: M– detection accuracy (mAP@50);
 V–total number of images in the dataset;
 V2–quadratic term representing the saturation effect;
 D–class imbalance level (ratio of the most to the least represented class);
 Ci– binary indicators for YOLOv10 configurations:
 C1 =YOLOv10-s,
 C2 =YOLOv10-m,
 C3 =YOLOv10-b,
 C4 =YOLOv10-l,
 C5 =YOLOv10-x;
 The baseline configuration is YOLOv10-n;
 ε–random error term.
 The proposed model allows for the quantitative evaluation of each factor’s contribution and was
 validated using three independent datasets (SH17, CHV, SHEL5K). The analysis confirmed the saturation
 effect: once the dataset reaches approximately 5,000 images, additional data yields only marginal
 improvements in accuracy.
 This model provides deeper insights into the key factors influencing PPE detection performance and
 supports the optimization of dataset preparation and balancing. It also assists in selecting the most effective
 model configurations, considering task-specific requirements and available computational resources.
