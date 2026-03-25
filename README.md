# NLW Operator: Computer Vision - Rocketseat 🚀

This repository contains the code developed during the **Computer Vision** track of the **Next Level Week (NLW) - Operator Python** program by **Rocketseat**.

A special highlight is that all the development in this project was carried out with the assistance of **Antigravity**, the AI assistant from Google DeepMind, which helped with both logic implementation and interface design.

---

## 📚 Program Journey

The project is divided into three main stages, following the order of the classes:

### 1. [LeNet](./lenet) - Digit Classification (MNIST)
In this first stage, we implemented the classic **LeNet-5** convolutional neural network architecture using **PyTorch**.
- **Goal**: Recognize handwritten digits from the MNIST dataset.
- **Highlights**: Modernization of the architecture with ReLU activation and Max Pooling, achieving over 98% accuracy.
- **Visualizations**: Exploration of convolution filters and network error analysis.

### 2. [Recog System](./recog_system) - Gesture Recognition System
In this stage, we advanced to custom gesture recognition and explored several state-of-the-art Computer Vision libraries.
- **Data Collection**: Script to capture hand landmarks via MediaPipe and save them to CSV.
- **Machine Learning**: Training a **Random Forest** classifier (Scikit-Learn) to recognize gestures such as "Rock", "Hang Loose", and "Small Heart".
- **CV Exploration**: Notebooks focused on:
  - Object detection (YOLOS).
  - Semantic segmentation (ClipSeg).
  - Image classification (TIMM).
  - Multimodal AI with **Gemini AI Vision**.

### 3. [Computer Vision App](./computer_vision_app) - Real-Time Web Application
The final stage consisted of creating a complete and high-performance web application to use the trained models.
- **Technologies**: Use of **FastHTML**, **WebSockets**, and **MediaPipe**.
- **Features**:
  - Low-latency video streaming.
  - Premium interface with **Glassmorphism** design.
  - Real-time controls (quality slider, landmark toggle).
  - "Match" system that detects the same gesture in both hands.

---

## 🛠️ Technologies Used

- **Language**: Python 3.12+
- **Deep Learning**: PyTorch, TIMM, ClipSeg, YOLOS.
- **Computer Vision**: OpenCV, MediaPipe.
- **Machine Learning**: Scikit-Learn, Pandas, NumPy.
- **Frontend/Backend**: FastHTML (FastAPI), WebSockets, Vanilla JS & CSS.
- **Package Management**: [uv](https://github.com/astral-sh/uv)

---

## 🚀 How to Run

This repository uses `uv` to manage dependencies. To set up the environment and run any of the sub-projects:

1. Install `uv` (if you don't have it yet):
   ```bash
   powershell -c "irm https://astral-sh/uv/install.ps1 | iex"
   ```

2. Synchronize dependencies in the desired directory:
   ```bash
   cd <directory-name>
   uv sync
   ```

3. Run the notebooks or scripts following the specific instructions in each folder's README.

---

Developed with 💜 during Rocketseat's NLW with the support of **Antigravity**.

