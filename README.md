# Signature Verification System
A deep learning-based signature verification system that combines convolutional neural networks (EfficientNet) with graph neural networks for highly accurate signature authenticity detection.
## overview
This system analyzes signature images to determine whether they are genuine or forged. It uses a hybrid approach that combines:

- Computer Vision: EfficientNet backbone for image feature extraction

- Graph Neural Networks: Analyzes signature stroke patterns and relationships

- Multi-Modal Fusion: Combines both visual and structural features for robust verification
## Key Features
Advanced Preprocessing: Adaptive histogram equalization, pressure simulation, and elastic deformations

Graph Representation: Converts signatures into graph structures with rich feature nodes

Hybrid Architecture: Combines CNN and GNN capabilities for superior performance

Focal Loss: Handles class imbalance between genuine and forged samples

Mixed Precision Training: Optimized for GPU performance
## Output Format
Verification results include:

genuine_prob: Probability the signature is genuine (0.0-1.0)

confidence: Confidence level in the prediction

verdict: Final classification (GENUINE/FORGED)

## Model Architecture
The hybrid model consists of three main components:

Vision Backbone: EfficientNetB0 for image feature extraction

Graph Network: Processes signature stroke relationships

Fusion Head: Combines visual and graph features for final classification

## Preprocessing Pipeline
Adaptive contrast enhancement (CLAHE)

Pressure simulation using distance transforms

Stroke width analysis

Elastic deformations (during augmentation)

Graph construction with multiple feature types:

Position and intensity

Gradient information

Curvature features

Stroke width

Texture features

## Performance
The system uses multiple metrics for evaluation:

Accuracy

AUC (Area Under Curve)

Precision

Recall

Focal Loss for handling class imbalance

## Configuration
Key configuration options in the code:

BASE_DIR: Root directory for the project

Image size: 650×650 pixels

Graph parameters: 1000 nodes, 3000 edges max

Training parameters: 100 epochs, early stopping

Verification threshold: 0.92 probability

## Requirements
Python 3.7+

TensorFlow 2.x

OpenCV

scikit-image

SciPy

scikit-learn

## Notes
The system is designed to work with white backgrounds and dark signatures

Input images should be in PNG format

Each person requires their own trained model

The system includes data augmentation for improved generalization

## Future Enhancements
Potential improvements:

Support for colored signatures

Real-time verification

Mobile deployment

Additional signature databases

Transfer learning across different writing styles
