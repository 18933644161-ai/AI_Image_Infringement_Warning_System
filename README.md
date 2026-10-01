# AI Image Infringement Warning System

A Visualized Prototype Based on High-Risk Facial Feature Database and Adaptive Interference


---

## Project Overview

This repository contains the web-based prototype for my paper published at ICAICN 2026. 

With the rapid advancement of Generative AI (e.g., Stable Diffusion, Deepfake), unauthorized generation of celebrity and public figures' portraits has become a severe legal and ethical issue. While traditional systems focus on passive detection, this project implements an Active Defense Framework. It not only detects high-risk similarities but also automatically applies adaptive perturbations to mitigate the infringement risk.

## Core Features (End-to-End Workflow)

1. User Input: Upload a test image via the Web UI.
2. Feature Extraction: Utilizes InsightFace (ArcFace) to extract 512-D facial feature embeddings.
3. Similarity Matching: Computes Cosine Similarity against a pre-built High-Risk Facial Feature Database (100 identities, 1,000+ features).
4. Adaptive Interference: If similarity exceeds the optimal threshold of 0.70, the system automatically applies dynamic perturbations (e.g., Gaussian Noise, Blur) to the facial region.
5. Re-evaluation and Reporting: Re-extracts features to calculate the "After Interference Similarity" and generates a visualized comparison chart.

## Representative Case Study

Note: Results vary depending on the input image and the selected interference method. Below is a representative high-risk case demonstrating the mitigation effect.

- Initial Similarity: 0.999 (HIGH Risk)
- After Interference: 0.639 (Safe Range)
- Similarity Reduction: ~36.0%
- Processing Time: ~0.57s (Near Real-time Inference)

## Key Trade-off and Future Direction

During the development of this prototype, I discovered a critical practical trade-off: while stronger interference successfully mitigates similarity to a safe range, it degrades visual image quality. 

This reveals that pure technical intervention is insufficient for real-world commercial deployment. It directly motivates my pursuit of cross-disciplinary studies at HKBU (MSc in AI and Digital Media)—to bridge the gap between technical algorithms, AI governance, and media ethics.

## Tech Stack

- Language: Python 3.11
- UI Framework: Gradio
- AI/Computer Vision: InsightFace, OpenCV, NumPy
- Data Visualization: Matplotlib
- Database: SQLite

## Quick Start and Note for Reviewers

**Note for Reviewers:** Due to the local environment setup complexity (including C++ compilation dependencies for InsightFace and large pre-trained model downloads), running the code locally on a new machine may require additional environment configuration. Please refer to the PDF Portfolio and Demo Video for a quick overview of the system's functionality and validation results.

If you still wish to run it locally:
1. Clone the repository.
2. Install dependencies: pip install -r requirements.txt
3. Run the application: python web_interface.py
4. Open http://127.0.0.1:8080 in your browser.
