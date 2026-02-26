# MNIST Image Recognition Neural Network

Vision is effortless for humans, but for a machine, recognizing a simple handwritten digit requires translating a chaotic matrix of 784 individual pixels into mathematical certainty. The challenge of this project was to engineer a neural network capable of generalizing handwritten patterns from the standardized MNIST database. 

To achieve this, an end-to-end deep learning classification module was developed utilizing the TensorFlow and Keras frameworks. The project implements an optimized baseline pipeline alongside an advanced experimental sandbox to evaluate how spatial visualization, architectural activation layers, and optimization algorithms impact structural convergence.

---

## Model Architecture & Pipeline Layout

The dataset underwent training and validation evaluation across 70,000 total digital image vectors. The execution is divided into two distinct engineering pipelines:

### 1. Production Baseline Pipeline (`1_mnist_baseline_classifier.py`)
Designed to establish the foundational architecture, proving out the data normalization, compilation, and evaluation logic required to achieve a highly accurate baseline:
* **Topology Configuration:** A robust hidden processing layer containing 128 interconnected nodes.
* **Activation Architecture:** Rectified Linear Unit (ReLU) deployed to maintain sparse activation profiles and mitigate vanishing gradient limitations, followed by a Softmax output distribution.
* **Optimization Framework:** Adam optimizer, utilizing adaptive learning rate moment estimations to accelerate convergence.
* **Empirical Performance:** Consistently achieved a baseline classification accuracy exceeding 97.00% on the withheld testing partition over a concise 5-epoch training cycle.

### 2. Experimental Evaluation Sandbox (`2_mnist_experimental_sandbox.py`)
Built as an advanced progression of the baseline, this module introduces targeted data exploration and visual validation tracking:
* **Spatial Search Logic:** Engineered custom matrix traversal to systematically isolate and visualize the first sequential index of every target class (digits 0-9).
* **Convergence Tracking:** Renders continuous validation trajectories via Matplotlib to visually diagnose model health, ensuring no overfitting occurs during the training epochs.
* **Architectural Experimentation:** Automates a structural comparison between modern ReLU networks and legacy Sigmoid configurations, definitively proving the optimization superiority of modern activation functions.

---

## Data Preprocessing & Feature Scaling

Prior to initializing forward propagation routines, the raw MNIST input space undergoes rigorous transformation steps:
1. **Dimension Vectorization:** Flattening structural 2D pixel matrices into a continuous 1D input array ($28 \times 28 = 784$ unique input dimensions per sample).
2. **MinMax Scaling Normalization:** Pixel intensity matrices range from absolute black ($0$) to peak white ($255$). These are rescaled programmatically to fit precisely within a uniform floating-point boundary:
   $$X_{\text{scaled}} = \frac{X - 0}{255 - 0} \in [0.0, 1.0]$$
   Normalizing these bounds protects the network from gradient saturation and ensures steady weight tracking across backpropagation cycles.

---

## Pipeline Initialization

To execute the neural network training sequences, activate your local virtual environment and launch the respective module.

```bash
.\.venv\Scripts\activate
python 1_mnist_baseline_classifier.py
python 2_mnist_experimental_sandbox.py