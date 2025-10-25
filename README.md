YOLOv11-Refine PPE Detection (CHV)

Paper: A PPE Detection Method Based on the Advanced YOLOv11 Architecture with a Three-Stage Refine Block System* (Sustainability, 2025)  
Dataset: CHV – Construction Helmet & Vest  https://github.com/ZijianWang1995/ppe_detection

Model Overview
YOLOv11-Refine introduces three refinement stages (R1–R3) to enhance multi-scale feature fusion and improve detection of small PPE elements (helmets, vests).  
Architecture: [`yolo11_refine.yaml`](./yolo11_refine.yaml)  
Compared to YOLOv11s, it achieves higher precision and mAP with minimal inference cost increase.

 Training
Framework: Ultralytics 8.3.54 (PyTorch 2.4.1)  
Epochs: 203 | Image size: 640 | Batch size: 16  
GPU: Quadro RTX 6000 ×4  

---

Results on CHV
(From [`CHVtrYolo11 r.txt`](./CHVtrYolo11%20r.txt))

| Model | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|:------|:-----------|:--------|:----------|:-------------|
| YOLOv11s | 0.867 | 0.812 | 0.856 | 0.496 |
| **YOLOv11s-Refine (R3)** | **0.894** | **0.809** | **0.873** | **0.508** |
