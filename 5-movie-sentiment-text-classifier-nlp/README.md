# Multi-Class Movie Sentiment Text Classifier

Human language is inherently chaotic. It is filled with nuance, slang, and unpredictable structures. For a machine to understand human emotion, it must first translate that chaos into mathematical certainty. The challenge of this project was to engineer a complete Natural Language Processing (NLP) deep learning pipeline capable of parsing unstructured movie reviews and categorically defining their underlying sentiment.

To achieve this, the system was designed to classify textual input into four absolute emotional states: Positive, Negative, Mixed, and Neutral. This required the implementation of rigorous tokenization, text vectorization, and global word embedding transformations to bridge the gap between human linguistics and machine intelligence.

---

## Data Engineering & Preprocessing

Raw text is useless to a neural network without rigorous preparation. To guarantee clean algorithmic convergence, a highly structured data engineering pipeline was established.

* **Corpus Sanitization:** Text assets were aggressively stripped of HTML markup and non-alphanumeric noise, then lowercased to safeguard vocabulary boundaries.
* **Tokenization & Sequence Padding:** The core of this transformation was the tokenization sequence. Character words were translated into unique numerical indexes. Because neural networks require uniform tensor shapes, arrays that fell short of the 120-word maximum length were padded using trailing post-zero padding tricks, maintaining strict multidimensional array consistency.

---

## Deep Learning NLP Architecture

Once the linguistic data was mathematically structured, a multi-layered Sequential neural network was deployed to decode the emotional context. The architecture transforms raw text data into multi-dimensional probability arrays through a precise sequence of layers:

* **Input Layer:** Ingests the tokenized integer text arrays.
* **Word Embedding Layer:** Maps discrete integer tokens into a continuous 32-dimensional geometric vector space (10,000 unique words $\times$ 32 dimensions). This is the brain of the operation, grouping semantically similar terms together so the machine can learn context.
* **Global Average Pooling:** Flattens variable spatial sequence lengths by averaging vector weights across the time dimension, preventing parameter explosions and maximizing computational efficiency.
* **Dense Hidden Processing:** A multi-layered dense network utilizing Rectified Linear Unit (ReLU) activations to extract and decode non-linear emotional patterns.
* **Output Classification:** Employs a Softmax activation function across four processing nodes to project definitive prediction probabilities across all sentiment classes.

---

## Project Layout

The repository is divided into three distinct pipeline phases:
* `1_sentiment_corpus_generator.py`: The foundational data engineering script that sanitizes text and generates synthetic training data to ensure class equilibrium.
* `2_dataset_balancing_stratifier.py`: The critical stratification layer that executes an 80/20 train/test split, preserving probability distributions.
* `3_embedding_network_trainer.py`: The terminal deep learning training and evaluation script utilizing TensorFlow and Keras.

---

## Local Deployment Setup

To initiate the natural language processing pipeline, activate your local virtual environment and execute the scripts sequentially.

```bash
.\.venv\Scripts\activate
python 1_sentiment_corpus_generator.py
python 2_dataset_balancing_stratifier.py
python 3_embedding_network_trainer.py