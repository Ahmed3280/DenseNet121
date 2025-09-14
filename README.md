# Waste Classification using DenseNet121 (PyTorch)

This repository contains a PyTorch implementation of a waste image classification model trained on the Waste Classification dataset: https://www.kaggle.com/datasets/techsash/waste-classification-data

- Uses a pretrained DenseNet121 backbone
- Initially, only the classifier was unfrozen, achieving good results:
  - Train Accuracy: ~91.25%
  - Validation Accuracy: ~93.46%
- To improve performance, DenseBlock 4 was unfrozen along with the classifier for fine-tuning, achieving the best results at epoch 10

Best Results (epoch 10):
- Train Accuracy: 95.49%
- Validation Accuracy: 95.21%
- Test Accuracy: 93.35%

Features:
- Supports progressive fine-tuning (classifier only → last dense block + classifier)
- Automatically saves best model weights based on validation accuracy
- Includes preprocessing, augmentation, and evaluation scripts

Tech Stack:
- Python 3.x
- PyTorch
- Torchvision (DenseNet121)
