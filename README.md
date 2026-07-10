# CodeAlpha_HandwrittenCharacterRecognition

**CodeAlpha Machine Learning Internship — Task 3**

## 📌 Objective
Identify handwritten digits using image processing and deep learning.

## 🧠 Approach
- Dataset: scikit-learn's `load_digits` — 1,797 8x8 grayscale images of handwritten digits (0-9), from the same UCI/NIST family of data as MNIST. Used so the project runs fully offline with no large download required.
- Model: **Convolutional Neural Network (CNN)** built with TensorFlow/Keras
  - 2 Conv2D layers + BatchNorm + MaxPooling + Dropout
  - Dense layers with dropout for regularization
  - EarlyStopping on validation loss
- Achieved **~99% test accuracy**

## 📂 Files
- `handwritten_recognition.py` — full pipeline: data loading, CNN model, training, evaluation, plotting
- `sample_digits.png`, `training_history.png`, `confusion_matrix.png` — generated after running the script
- `handwritten_digit_cnn.keras` — the saved trained model

## ▶️ How to Run
```bash
pip install numpy scikit-learn matplotlib tensorflow
python handwritten_recognition.py
```

## 📈 Results (example run)
- **Test Accuracy: 98.9%**
- **Test Loss: 0.027**
- Per-class precision/recall/F1 all ≥ 0.94, most classes at 1.00

## 🔁 Scaling Up to Full MNIST / EMNIST
This project uses 8x8 `load_digits` for a fast, fully-offline demo. To use the full 28x28 MNIST dataset or EMNIST (letters), swap the loader:
```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
```
The same CNN architecture (with input_shape adjusted to `(28, 28, 1)`) works unchanged, and can be extended toward full word/sentence recognition with a CRNN (CNN + RNN/LSTM) for sequence modeling, as suggested in the task brief.

## 🎥 Submission Checklist (CodeAlpha)
- [ ] Push this repo to GitHub as `CodeAlpha_HandwrittenCharacterRecognition`
- [ ] Record a short video walkthrough and post it on LinkedIn tagging @CodeAlpha, with the GitHub link
- [ ] Submit via the CodeAlpha submission form
