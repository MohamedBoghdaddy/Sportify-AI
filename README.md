# 🏋️‍♂️🏅 **Sportify AI**: Intelligent Sports Image Classifier

## 📄 Project Description

**Sportify AI** is a cutting-edge deep learning project designed to classify sports images into six distinct categories. Leveraging both a custom-built Convolutional Neural Network (CNN) and the power of MobileNetV2 transfer learning, Sportify AI ensures high accuracy and efficiency for automated sports analysis tasks.

### Supported Sports Categories
- 🏀 Basketball
- ⚽ Football
- 🚣 Rowing
- 🏊 Swimming
- 🎾 Tennis
- 🧘 Yoga

---

## 📂 Dataset

### Dataset Structure
The dataset is organized into **training** and **testing** directories:
- **Training Directory:** Contains labeled images for training and validation.
- **Testing Directory:** Contains unlabeled images for predictions.

### Characteristics
- **Image Formats:** `.jpg`, `.jpeg`, `.png`
- **Variable Dimensions:** Automatically preprocessed to uniform dimensions

---

## 🚀 Features

- **Custom CNN Model:** Built from scratch to learn sports-specific features.
- **Transfer Learning:** Fine-tuned MobileNetV2 model for improved accuracy and reduced training time.
- **Batch Predictions:** Efficiently processes batches of test images for quick predictions.
- **Metrics Visualization:** Real-time training accuracy and loss visualization.
- **Adaptable Design:** Easily extendable to classify other image datasets.

---

## 🛠️ Tech Stack

- **Framework:** TensorFlow, Keras
- **Programming Language:** Python
- **Visualization Tools:** Matplotlib, Pandas
- **Pre-trained Model:** MobileNetV2
- **Environment:** Google Colab / Local Environment

---

## 📋 Installation and Setup

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Jupyter Notebook or Google Colab

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sportify-ai.git
   cd sportify-ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - **Training Data:** Place images in `dataset/Train/`.
   - **Test Data:** Place images in `dataset/Test/`.

4. Run the scripts:
   - Training: `train.py`
   - Evaluation: `evaluate.py`
   - Testing: `test.py`

---

## 🖼️ Model Training and Prediction

### Training
1. **Train Custom CNN Model:**
   ```bash
   python train.py
   ```

2. **Train MobileNetV2 Pre-trained Model:**
   ```bash
   python train.py
   ```

3. Save model weights:
   - `cnn_model_weights.h5`
   - `pretrained_model_weights.h5`

### Testing
1. Load test images from `dataset/Test/`.
2. Generate predictions:
   ```bash
   python test.py
   ```
3. Results are saved in `predictions.csv`.

---

## 📊 Visualization

- **Training Logs:** Displayed during training for real-time monitoring.
- **Accuracy/Loss Graphs:** Plotted after training to analyze model performance.
- **Prediction Results:** CSV file (`predictions.csv`) contains filenames and predicted labels.

---

## 📈 Results

| **Model**           | **Validation Accuracy** | **Testing Accuracy** |
|---------------------|-------------------------|-----------------------|
| Custom CNN          | xx.xx%                 | xx.xx%               |
| MobileNetV2         | xx.xx%                 | xx.xx%               |

---

## 📌 Future Enhancements

- Apply advanced data augmentation techniques for robustness.
- Explore additional pre-trained models like EfficientNet and ResNet.
- Deploy the classifier as a web application for real-time predictions.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 💡 Acknowledgments

- TensorFlow and Keras Community
- [ImageNet](https://www.image-net.org/) for pre-trained weights
- Open-source contributors for inspiration and support
