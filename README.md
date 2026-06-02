# Haze-Removal-and-Real-time-Object-Detection-using-a-Hybrid-Deep-Learning-Framework
# Haze Removal and Real-Time Object Detection using a Hybrid Deep Learning Framework

## Overview

Poor weather conditions such as haze, fog, rain, and low-light environments significantly degrade image quality, reducing the effectiveness of surveillance and object detection systems.

This project presents a hybrid deep learning framework that combines advanced image enhancement techniques with real-time object detection. The framework first restores degraded images using Latent Diffusion Models (LDM), Scale-Wise Distillation (SWD), and Patch-Based Diffusion, then performs object detection using YOLOv8-Nano.

The proposed system is designed for intelligent surveillance, autonomous vehicles, drones, and smart city applications where visibility is often compromised.

---

## Key Features

- Multi-scale image enhancement using Latent Diffusion Models
- Scale-Wise Distillation (SWD) for faster inference
- Patch-Based Diffusion for preserving fine image details
- Feature-level fusion of global and local image features
- Real-time object detection using YOLOv8-Nano
- End-to-end enhancement and detection pipeline
- Optimized for deployment on resource-constrained devices

---

## System Architecture

```text
Input Image / Video Frame
            │
            ▼
      Preprocessing
            │
            ▼
      VAE Encoder
            │
            ▼
    Latent Representation
            │
 ┌──────────┼──────────┐
 │          │          │
 ▼          ▼          ▼
Multi-    Scale-    Patch-
Scale     Wise      Based
Diffusion Distill.  Diffusion
 │          │          │
 └──────────┼──────────┘
            ▼
     Feature Fusion
            │
            ▼
      VAE Decoder
            │
            ▼
 Enhanced Image
            │
            ▼
      YOLOv8-Nano
            │
            ▼
 Object Detection Output
