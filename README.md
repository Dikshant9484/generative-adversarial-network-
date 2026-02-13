# 🎨 CIFAR-10 DCGAN Image Generator (TensorFlow)

A Deep Convolutional Generative Adversarial Network (DCGAN) project that learns the distribution of CIFAR-10 images and generates new synthetic images using TensorFlow/Keras. The model is trained adversarially using a Generator and Discriminator network.

---

## 🚀 Project Overview

This project implements a GAN that generates 32×32 RGB images similar to the CIFAR-10 dataset. The Generator creates fake images from random noise, while the Discriminator learns to distinguish real images from generated ones. Through adversarial training, the Generator improves its image quality over time.

This project is suitable for:

* Deep Learning coursework
* GAN fundamentals practice
* Resume ML projects
* Image generation demos

---

## 🧠 Model Architecture

### Generator

* Input: Random noise vector (latent_dim = 100)
* Dense → BatchNorm → LeakyReLU
* Reshape to feature maps
* Conv2DTranspose upsampling layers
* Output: 32×32×3 RGB image (tanh activation)

### Discriminator

* Input: 32×32×3 image
* Conv2D downsampling layers
* LeakyReLU + Dropout
* Flatten + Dense
* Output: Real/Fake score

---

## 📦 Dataset

**CIFAR-10**

* 60,000 color images
* Size: 32×32
* 10 object classes
* Automatically downloaded via TensorFlow

Dataset is normalized to range **[-1, 1]** for GAN training compatibility.

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* VS Code or Google Colab

---

## 🛠 Installation

Create virtual environment (recommended):

```bash
python -m venv ganenv
ganenv\Scripts\activate
```

Install dependencies:

```bash
pip install tensorflow numpy matplotlib
```

---

## ▶️ How to Run

```bash
python main.py
```

Training will begin and print epoch progress. After training completes, generated images will be displayed.

---

## 🔧 Key Hyperparameters

| Parameter     | Value                   |
| ------------- | ----------------------- |
| Batch Size    | 128 (reduce if low RAM) |
| Latent Dim    | 100                     |
| Epochs        | 50+ recommended         |
| Optimizer     | Adam                    |
| Learning Rate | 1e-4                    |

Low-memory systems can use:

```python
BATCH_SIZE = 32
BUFFER_SIZE = 8000
```

---

## 🖼 Output

* Generator produces synthetic CIFAR-like images
* Early epochs → noisy/blurry
* Later epochs → more structured objects
* Image quality improves with more epochs and GPU training

---

## ⚠️ Limitations

* This is an **unconditional GAN**
* Cannot generate images from text prompts
* No class control unless upgraded to Conditional GAN
* CIFAR resolution limits visual sharpness
* CPU training is slow

---

## 🚀 Possible Improvements

* Conditional GAN (class-controlled output)
* Save generated samples per epoch
* Add loss tracking plots
* Deploy generator as API
* Build web UI for generation
* Train on higher resolution dataset
* Upgrade to StyleGAN / Diffusion models

---

## 🎓 Learning Outcomes

This project demonstrates:

* GAN adversarial training
* Generator vs Discriminator dynamics
* TensorFlow model building
* tf.data pipeline usage
* Image normalization for GANs
* Batch training mechanics

---

## 🧾 License

Educational and research use.

---

## 👨‍💻 Author

GAN Image Generation Project — TensorFlow DCGAN Implementation
