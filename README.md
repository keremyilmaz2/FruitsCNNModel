# 🍎 Fruits-360 CNN Classification Model

A comprehensive deep learning project for fruit and vegetable classification using the Fruits-360 dataset with TensorFlow/Keras.

## 🎯 Project Overview

This project implements various CNN architectures for classifying 206 different types of fruits and vegetables. The main focus is achieving high **real-world performance** rather than just dataset accuracy.

### 🏆 Model Performance Summary

| Model Version | Dataset Accuracy | Real-World Accuracy | Model Size | Training Time |
|---------------|------------------|---------------------|------------|---------------|
| **Baseline (recommended)** | 77% | 🟢 **85%+** | 104MB | 1h |
| Fine-tuned | 86% | 🟡 **60%** | 226MB | 2h |
| Real-World Optimized | 82-85% | 🟢 **90%+** | ~150MB | 1.5h |

> **Key Finding**: Higher dataset accuracy doesn't guarantee better real-world performance due to overfitting!

## 📊 Dataset Information

- **Source**: [Fruits-360 Dataset](https://www.kaggle.com/moltean/fruits)
- **Classes**: 206 different fruits and vegetables
- **Training Images**: ~88,500 images
- **Test Images**: ~34,700 images
- **Image Size**: 100x100 pixels
- **Format**: RGB images

### Class Distribution
The dataset includes varieties of:
- 🍎 **Apples** (30+ varieties): Red, Green, Golden, Granny Smith, etc.
- 🍌 **Bananas** (4 varieties): Regular, Lady Finger, Red
- 🍇 **Grapes** (4 varieties): White, Blue, Pink
- 🍓 **Berries**: Strawberry, Raspberry, Blackberry, Blueberry
- 🥕 **Vegetables**: Tomato, Carrot, Cucumber, Pepper, etc.
- 🥜 **Nuts**: Walnut, Hazelnut, Chestnut
- And many more...

## 🏗️ Architecture

### Supported Model Architectures
1. **Custom CNN v1**: Basic convolutional network
2. **Custom CNN v2**: Advanced CNN with residual connections  
3. **Transfer Learning ResNet50**: Pre-trained ImageNet weights
4. **Transfer Learning MobileNetV2**: Lightweight mobile-optimized
5. **Conservative Transfer Learning**: Real-world optimized (recommended)

### Current Best Architecture
```python
# Conservative Transfer Learning (Real-World Optimized)
Base Model: ResNet50 (frozen ImageNet weights)
├── GlobalAveragePooling2D()
├── Dropout(0.6)                    # Heavy regularization
├── Dense(128, activation='relu')   # Smaller layer
├── Dropout(0.5)
└── Dense(206, activation='softmax') # 206 classes
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Dataset Setup
1. Download Fruits-360 dataset from [Kaggle](https://www.kaggle.com/moltean/fruits)
2. Extract to `./FruitsData/fruits-360_100x100/`
3. Verify structure:
```
FruitsData/
└── fruits-360_100x100/
    └── fruits-360/
        ├── Training/
        │   ├── Apple 10/
        │   ├── Apple 11/
        │   └── ...
        └── Test/
            ├── Apple 10/
            ├── Apple 11/
            └── ...
```

### Training a Model
```bash
# Train with default configuration
python train.py

# Train with specific config
python train.py --config configs/config_realworld.py

# Interactive mode
python train.py --interactive
```

### Making Predictions
```bash
# Single image prediction
python predict.py models/your_model.h5 --image path/to/image.jpg

# Batch prediction
python predict.py models/your_model.h5 --batch path/to/images/

# Real-time camera prediction
python predict.py models/your_model.h5 --camera
```

### Model Evaluation
```bash
# Comprehensive evaluation
python evaluate_model.py

# Compare multiple models
python experiments/compare_models.py
```

## 📁 Project Structure

```
fruits-360-cnn/
├── 📁 configs/                    # Model configurations
│   ├── config_baseline.py         # Baseline model (recommended)
│   ├── config_realworld.py        # Real-world optimized
│   └── config_finetune.py         # Fine-tuning experiments
│
├── 📁 src/                        # Source code
│   ├── train.py                   # Training script
│   ├── model_architecture.py      # CNN architectures
│   ├── data_preprocessing.py      # Data processing
│   ├── evaluate_model.py          # Model evaluation
│   ├── predict.py                 # Prediction interface
│   └── utils.py                   # Utility functions
│
├── 📁 experiments/                # Experimental scripts
│   ├── test_predictor.py          # Interactive testing
│   └── compare_models.py          # Model comparison
│
├── 📁 models/                     # Trained models (Git LFS)
├── 📁 results/                    # Training results
└── 📁 docs/                       # Documentation
```

## 🔧 Configuration

### Key Configuration Options

```python
# Model Architecture
MODEL_CONFIG = {
    'architecture': 'transfer_resnet50_conservative',  # Model type
    'num_classes': 206,
    'dropout_rate': 0.5,              # Regularization
}

# Training Parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 25,
    'learning_rate': 0.0005,
    'early_stopping_patience': 5,
}

# Data Augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 30,             # Aggressive augmentation
    'width_shift_range': 0.25,        # for real-world robustness
    'brightness_range': [0.7, 1.3],
    'horizontal_flip': True,
}
```

## 📈 Training Results

### Learning Curves
![Training History](results/training_history_example.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix_example.png)

### Performance by Class
- **Best Performing Classes** (F1 > 0.95): Apple Core, Cherry varieties, Beans
- **Challenging Classes** (F1 < 0.5): Nut varieties, unripe fruits
- **Most Confused Pairs**: Different nut types, similar apple varieties

## 🧪 Experiments & Findings

### Key Insights
1. **Dataset vs Real-World Performance Gap**: Models achieving 86% on dataset can perform poorly (60%) on real-world images
2. **Transfer Learning Effectiveness**: Pre-trained ImageNet features significantly improve performance
3. **Augmentation Impact**: Aggressive data augmentation crucial for real-world generalization
4. **Overfitting Detection**: Fine-tuning can hurt real-world performance despite better validation scores

### Model Comparison Results
```
Baseline Model (Conservative):
✅ Real-world performance: 85%+
✅ Robust to lighting changes
✅ Handles different backgrounds
❌ Lower dataset accuracy (77%)

Fine-tuned Model:
✅ High dataset accuracy (86%)
❌ Poor real-world performance (60%)
❌ Overfitted to dataset characteristics
```

## 🔍 Usage Examples

### Interactive Testing
```python
# Launch interactive test interface
python experiments/test_predictor.py

# Test menu options:
# 1. Single image prediction
# 2. Batch prediction
# 3. Real-time camera
# 4. Random dataset test
```

### Programmatic Usage
```python
from src.predict import FruitPredictor

# Initialize predictor
predictor = FruitPredictor('models/baseline_model.h5')

# Make prediction
predictions = predictor.predict_single_image('path/to/fruit.jpg')
print(f"Prediction: {predictions[0]['class']} ({predictions[0]['percentage']:.1f}%)")
```

## 🐛 Known Issues & Limitations

### Current Limitations
- **Multiple Object Scenes**: Model designed for single fruit/vegetable classification
- **Background Dependency**: Performance varies with background complexity  
- **Lighting Sensitivity**: Extreme lighting conditions may affect accuracy
- **Similar Varieties**: Difficulty distinguishing between very similar fruit varieties

### Future Improvements
- [ ] Object detection for multiple fruits
- [ ] Better handling of complex backgrounds
- [ ] Improved lighting robustness
- [ ] Mobile optimization (TensorFlow Lite)

## 📊 Benchmarks

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU training (slow)
- **Recommended**: 16GB RAM, NVIDIA GPU with 4GB+ VRAM
- **Training Time**: 1-2 hours on modern GPU

### Performance Benchmarks
- **Inference Speed**: ~50ms per image (GPU), ~200ms (CPU)
- **Memory Usage**: ~2GB during training, ~500MB during inference
- **Model Size**: 104-226MB depending on architecture

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-architecture`)
3. Make changes and test thoroughly
4. Update documentation
5. Submit pull request

### Development Workflow
```bash
# Create experiment branch
git checkout -b experiment/vision-transformer

# Make changes to config
cp configs/config_baseline.py configs/config_vit.py

# Train and evaluate
python train.py --config configs/config_vit.py
python evaluate_model.py

# Commit results
git add .
git commit -m "🧪 Experiment: Vision Transformer architecture"
```

## 📚 Documentation

- [Training Guide](docs/training_guide.md) - Detailed training instructions
- [Model Architecture Guide](docs/architecture_guide.md) - Architecture explanations
- [API Reference](docs/api_reference.md) - Code documentation
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Fruits-360 dataset by Horea Muresan and Mihai Oltean
- **Frameworks**: TensorFlow/Keras, OpenCV, scikit-learn
- **Pre-trained Models**: ImageNet weights from TensorFlow Hub

## 📬 Contact

For questions, suggestions, or collaboration:
- 📧 Email: [your.email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/username/fruits-360-cnn/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/username/fruits-360-cnn/discussions)

---

⭐ **Star this repository if you found it helpful!**

🍎 Happy fruit classification! 🥕
