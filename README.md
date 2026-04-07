# Natural-Language-Proccessing-LSTM-vs-Vanilla-RNN



## 1. Project Overview
This project implements and evaluates two Recurrent Neural Network (RNN) architectures using **PyTorch** for the task of Fake News Detection.The assignment focuses on comparing a standard RNN with a Long Short-Term Memory (LSTM) network to observe how gating mechanisms address the vanishing gradient problem in NLP.

## 2. Dataset
The project utilizes the **Fake News Dataset** To run the implementation, ensure the following files are in the root directory:
* `True.csv`: Contains legitimate news articles.
* `Fake.csv`: Contains fabricated news articles.
The dataset used for this project can be downloaded from Kaggle at the following link:
[Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data)

Please ensure both `True.csv` and `Fake.csv` from this download are placed in the project root directory before execution.

## 3. Requirements
The following Python libraries are required to execute the source code:
* `torch` (PyTorch)
* `pandas`
* `scikit-learn`
* `numpy`

Install dependencies using:
```bash
pip install torch pandas scikit-learn numpy
```

##  Architectural Comparison: Vanilla RNN vs. LSTM

This section documents the technical differences between the two models implemented in this project, specifically addressing why the LSTM is the superior choice for Natural Language Processing tasks.

### 1. Vanilla Recurrent Neural Network (RNN)
The Vanilla RNN is the foundational architecture for sequential data. It processes inputs ($x_t$) and maintains a hidden state ($h_t$) that is updated at every time step.

* **Mathematical Bottleneck**: In a standard RNN, the gradient must be multiplied by the same weight matrix $W$ repeatedly during Backpropagation Through Time (BPTT).
* **The Vanishing Gradient Problem**: If the weights are small, the gradient shrinks exponentially as it travels back to earlier time steps. This is represented by the product term derived in the Homework 9 proof: $\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$.
* **Result**: The model "forgets" information from the beginning of a long news article by the time it reaches the end, making it difficult to capture long-term dependencies.



---

### 2. Long Short-Term Memory (LSTM)
The LSTM was introduced specifically to mitigate the vanishing gradient problem found in Vanilla RNNs.

* **The Cell State ($C_t$)**: Unlike the RNN, which only has a hidden state, the LSTM introduces a "cell state" that acts as a high-speed conveyor belt for information. This allows the gradient to flow through the sequence with minimal interference.
* **Gating Mechanisms**: The LSTM uses three "gates" to selectively update the cell state:
    * **Forget Gate**: Decides which information from the previous state is no longer relevant.
    * **Input Gate**: Determines which new information from the current word should be stored.
    * **Output Gate**: Controls which part of the internal memory is sent to the next hidden state.
* **Result**: By protecting the cell state, the LSTM can maintain a stable gradient over long sequences, allowing it to "remember" context from the start of a document to the end.



---

### Performance Summary

| Feature | Vanilla RNN | LSTM |
| :--- | :--- | :--- |
| **Complexity** | 1 Activation Layer ($tanh$)  | 4 Interacting Layers (Gates)  |
| **Memory Capacity** |Short-term; fades over time  | Long-term; managed by Cell State  |
| **BPTT Stability** | Prone to Vanishing Gradients  | Stable; preserves gradient flow  |
| **NLP Suitability** | Short sentences/sequences  | Long-form articles (Fake News)  |
