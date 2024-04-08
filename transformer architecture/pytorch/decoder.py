# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Diving into the Transformer - Building your own Foundation LLM
# MAGIC
# MAGIC This lesson introduces the underlying structure of transformers from token management to the layers in a decoder, to comparing smaller and larger models.
# We will build up all of the steps needed to create our foundation model before training. You will see how the layers are constructed, and how the next word is chosen.
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Identify the key structures and functions in decoder transformers
# MAGIC 1. Analyze the effect of hyperparameter changes (such as embedding dimension) on the size of the LLM
# MAGIC 1. Compare the different performance of models with different model architectures

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# Import the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# COMMAND ----------

# MAGIC %md # Section 1: Encoding Natural Language - Word Embedding and Positional Encoding
# MAGIC
# MAGIC In this section we'll look at how to take a natural language input and convert it to the form we'll need for our transformer.

# COMMAND ----------

# Define a sentence and a simple word2id mapping
sentence = "The quick brown fox jumps over the lazy dog"
word2id = {word: i for i, word in enumerate(set(sentence.split()))}
print(
    word2id
)  # {'dog': 0, 'The': 1, 'over': 2, 'the': 3, 'fox': 4, 'quick': 5, 'jumps': 6, 'brown': 7, 'lazy': 8}

# Convert text to indices
input_ids = torch.tensor([word2id[word] for word in sentence.split()])
print(input_ids)  # tensor([1, 5, 7, 4, 6, 2, 3, 8, 0])

input_ids.max()  # tensor(8)


# Define a simple word embedding function
# https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
def get_word_embeddings(input_ids, embedding_size):
    embedding_layer = nn.Embedding(input_ids.max() + 1, embedding_size)
    return embedding_layer(input_ids)


# Get word embeddings
embedding_size = 16  # Size of the word embeddings
word_embeddings = get_word_embeddings(input_ids, embedding_size)
print(word_embeddings)
"""
tensor([[ 4.8642e-01, -1.2103e+00, -2.0510e+00, -9.1600e-01,  6.7203e-01,
          2.3728e-01, -1.2434e+00,  7.8907e-01, -9.2641e-01, -4.5371e-04,
         -3.5691e-01,  1.4008e-01,  1.3020e+00,  5.1590e-01,  7.9118e-02,
         -4.1217e-01],
        [-1.2736e+00,  3.7654e-01,  4.6414e-01,  1.6239e-02, -5.0390e-01,
          2.1713e+00, -4.8524e-01, -1.0866e+00,  8.7959e-01,  2.5965e-02,
          7.1266e-01,  1.6850e+00,  7.5564e-01, -5.6978e-01, -1.9154e+00,
          5.2040e-02],
        [-4.1068e-01, -9.1978e-01,  1.1046e+00,  8.9580e-01, -1.7582e+00,
          1.0378e+00, -5.8617e-03,  1.9077e-01,  6.9195e-01, -1.1683e+00,
         -7.1256e-01,  3.3992e-01,  3.2923e-01, -1.0721e+00,  6.8281e-01,
          6.5096e-01],
        [ 4.0923e-01,  6.4336e-01, -1.9461e+00,  1.5387e-03,  9.8465e-01,
         -2.6624e-01,  9.2657e-01, -3.9879e-01,  7.7525e-01, -2.9851e-01,
         -1.1788e+00,  1.3230e-02,  2.1367e+00, -2.1453e-01,  1.0875e+00,
          1.4436e+00],
        [-5.2008e-01,  1.2559e-01, -2.2351e-01,  3.7570e-02, -1.0538e+00,
          7.4769e-01, -8.5345e-01, -4.3269e-01, -1.0342e+00,  8.3754e-01,
         -8.7643e-01, -9.0855e-01, -8.2738e-01, -8.3519e-01,  1.9850e-01,
         -2.9305e-02],
        [-5.8878e-01, -1.0046e+00,  3.5360e-01,  1.1876e+00, -2.3145e-01,
          3.5064e-01,  1.4999e+00,  1.6554e+00, -6.5632e-01, -2.9200e+00,
          2.1077e-01,  3.8568e-02, -6.9371e-01, -2.2625e-01, -8.2806e-01,
          1.3337e+00],
         -1.2689e+00, -1.0933e+00,  1.2823e+00,  6.7278e-03, -4.2241e-01,
         -2.1517e+00, -7.9408e-01,  4.8980e-01, -5.9775e-02, -1.2684e+00,
          7.3545e-01],
        [-1.4415e-01, -8.8871e-01,  1.9124e+00, -6.5834e-01, -4.6082e-01,
          1.0175e+00, -1.9804e-01, -4.8014e-01, -1.1545e+00, -9.0472e-03,
          2.8593e-01,  2.5557e-01, -7.8646e-01, -9.1404e-01, -9.6197e-01,
          1.3965e+00],
        [ 1.0101e+00,  6.1960e-01,  2.1711e-02,  1.6540e+00, -2.4938e-02,
         -1.0210e-01,  1.2995e+00,  3.8193e-02,  8.9050e-02,  1.4404e+00,
          2.2739e-01, -1.9346e+00, -5.7790e-01, -1.4886e+00,  2.6249e-01,
         -3.5698e-02]], grad_fn=<EmbeddingBackward0>) """

# COMMAND ----------


# Define a function to generate positional encodings
def get_positional_encoding(max_seq_len, d_model):
    position = np.arange(max_seq_len)[:, np.newaxis]
    print(f"position: {position}, shape: {position.shape}")
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    print(f"div_term: {div_term}, shape: {div_term.shape}")

    positional_encoding = np.zeros((max_seq_len, d_model))
    print(
        f"positional_encoding: {positional_encoding}, shape: {positional_encoding.shape}"
    )

    positional_encoding[:, 0::2] = np.sin(position * div_term)
    print(
        f"positional_encoding[:, 0::2]: {positional_encoding[:, 0::2]}, shape: {positional_encoding[:, 0::2].shape}"
    )

    positional_encoding[:, 1::2] = np.cos(position * div_term)
    print(
        f"positional_encoding[:, 1::2] : {positional_encoding[:, 1::2] }, shape: {positional_encoding[:, 1::2].shape}"
    )

    print(
        f"torch.tensor(positional_encoding, dtype=torch.float): {torch.tensor(positional_encoding, dtype=torch.float)}, shape: {torch.tensor(positional_encoding, dtype=torch.float).shape}"
    )

    return torch.tensor(positional_encoding, dtype=torch.float)


# COMMAND ----------


# Function to plot heatmap
# ------------------------
def plot_heatmap(data, title):
    plt.figure(figsize=(5, 5))
    seaborn.heatmap(data, cmap="cool", vmin=-1, vmax=1)
    plt.ylabel("Word/token")
    plt.xlabel("Positional Encoding Vector")
    plt.title(title)
    plt.show()


# Generate and plot positional encoding
# -------------------------------------
# Get positional encodings
max_seq_len = len(sentence.split())  # Maximum sequence length 9
d_model = embedding_size  # Same as the size of the word embeddings 16
positional_encodings = get_positional_encoding(max_seq_len, d_model)
"""
position: [[0]
 [1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]], shape: (9, 1)
div_term: [1.00000000e+00 3.16227766e-01 1.00000000e-01 3.16227766e-02
 1.00000000e-02 3.16227766e-03 1.00000000e-03 3.16227766e-04], shape: (8,)
positional_encoding: [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]], shape: (9, 16)
positional_encoding[:, 0::2]: [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
 [ 8.41470985e-01  3.10983593e-01  9.98334166e-02  3.16175064e-02
   9.99983333e-03  3.16227239e-03  9.99999833e-04  3.16227761e-04]
 [ 9.09297427e-01  5.91127117e-01  1.98669331e-01  6.32033979e-02
   1.99986667e-02  6.32451316e-03  1.99999867e-03  6.32455490e-04]
 [ 1.41120008e-01  8.12648897e-01  2.95520207e-01  9.47260913e-02
   2.99955002e-02  9.48669068e-03  2.99999550e-03  9.48683156e-04]
 [-7.56802495e-01  9.53580740e-01  3.89418342e-01  1.26154067e-01
   3.99893342e-02  1.26487733e-02  3.99998933e-03  1.26491073e-03]
 [-9.58924275e-01  9.99946517e-01  4.79425539e-01  1.57455898e-01
   4.99791693e-02  1.58107295e-02  4.99997917e-03  1.58113817e-03]
 [-2.79415498e-01  9.47148158e-01  5.64642473e-01  1.88600287e-01
   5.99640065e-02  1.89725276e-02  5.99996400e-03  1.89736546e-03]
 [ 6.56986599e-01  8.00421646e-01  6.44217687e-01  2.19556091e-01
   6.99428473e-02  2.21341359e-02  6.99994283e-03  2.21359255e-03]
 [ 9.89358247e-01  5.74317769e-01  7.17356091e-01  2.50292358e-01
   7.99146940e-02  2.52955229e-02  7.99991467e-03  2.52981943e-03]], shape: (9, 8)
positional_encoding[:, 1::2] : [[ 1.          1.          1.          1.          1.          1.
   1.          1.        ]
 [ 0.54030231  0.95041528  0.99500417  0.99950004  0.99995     0.999995
   0.9999995   0.99999995]
 [-0.41614684  0.80657841  0.98006658  0.99800067  0.99980001  0.99998
   0.999998    0.9999998 ]
 [-0.9899925   0.58275361  0.95533649  0.99550337  0.99955003  0.999955
   0.9999955   0.99999955]
 [-0.65364362  0.30113746  0.92106099  0.99201066  0.99920011  0.99992
   0.999992    0.9999992 ]
 [ 0.28366219 -0.01034232  0.87758256  0.98752602  0.99875026  0.999875
   0.9999875   0.99999875]
 [ 0.96017029 -0.32079646  0.82533561  0.98205394  0.99820054  0.99982001
   0.999982    0.9999982 ]
 [ 0.75390225 -0.59943739  0.76484219  0.97559988  0.997551    0.99975501
   0.9999755   0.99999755]
 [-0.14550003 -0.81863246  0.69670671  0.9681703   0.99680171  0.99968002
   0.999968    0.9999968 ]], shape: (9, 8)
torch.tensor(positional_encoding, dtype=torch.float): tensor([[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,
          0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          1.0000e+00],
        [ 8.4147e-01,  5.4030e-01,  3.1098e-01,  9.5042e-01,  9.9833e-02,
          9.9500e-01,  3.1618e-02,  9.9950e-01,  9.9998e-03,  9.9995e-01,
          3.1623e-03,  9.9999e-01,  1.0000e-03,  1.0000e+00,  3.1623e-04,
          1.0000e+00],
        [ 9.0930e-01, -4.1615e-01,  5.9113e-01,  8.0658e-01,  1.9867e-01,
          9.8007e-01,  6.3203e-02,  9.9800e-01,  1.9999e-02,  9.9980e-01,
          6.3245e-03,  9.9998e-01,  2.0000e-03,  1.0000e+00,  6.3246e-04,
          1.0000e+00],
        [ 1.4112e-01, -9.8999e-01,  8.1265e-01,  5.8275e-01,  2.9552e-01,
          9.5534e-01,  9.4726e-02,  9.9550e-01,  2.9996e-02,  9.9955e-01,
          9.4867e-03,  9.9995e-01,  3.0000e-03,  1.0000e+00,  9.4868e-04,
          1.0000e+00],
        [-7.5680e-01, -6.5364e-01,  9.5358e-01,  3.0114e-01,  3.8942e-01,
          9.2106e-01,  1.2615e-01,  9.9201e-01,  3.9989e-02,  9.9920e-01,
          1.2649e-02,  9.9992e-01,  4.0000e-03,  9.9999e-01,  1.2649e-03,
          1.0000e+00],
        [-9.5892e-01,  2.8366e-01,  9.9995e-01, -1.0342e-02,  4.7943e-01,
          8.7758e-01,  1.5746e-01,  9.8753e-01,  4.9979e-02,  9.9875e-01,
          1.5811e-02,  9.9988e-01,  5.0000e-03,  9.9999e-01,  1.5811e-03,
          1.0000e+00],
        [-2.7942e-01,  9.6017e-01,  9.4715e-01, -3.2080e-01,  5.6464e-01,
          8.2534e-01,  1.8860e-01,  9.8205e-01,  5.9964e-02,  9.9820e-01,
          1.8973e-02,  9.9982e-01,  6.0000e-03,  9.9998e-01,  1.8974e-03,
        [ 6.5699e-01,  7.5390e-01,  8.0042e-01, -5.9944e-01,  6.4422e-01,
          7.6484e-01,  2.1956e-01,  9.7560e-01,  6.9943e-02,  9.9755e-01,
          2.2134e-02,  9.9976e-01,  6.9999e-03,  9.9998e-01,  2.2136e-03,
          1.0000e+00],
        [ 9.8936e-01, -1.4550e-01,  5.7432e-01, -8.1863e-01,  7.1736e-01,
          6.9671e-01,  2.5029e-01,  9.6817e-01,  7.9915e-02,  9.9680e-01,
          2.5296e-02,  9.9968e-01,  7.9999e-03,  9.9997e-01,  2.5298e-03,
          1.0000e+00]]), shape: torch.Size([9, 16]) """


plot_heatmap(positional_encodings, "Positional Encoding")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpreting the Positional Encoding Map
# MAGIC In the Transformer model, positional encoding is used to give the model some information about the relative positions of the words in the sequence since the
# Transformer does not have any inherent sense of order of the input sequence.
# MAGIC
# MAGIC The positional encoding for a position \(p\) in the sequence and a dimension \(i\) in the embedding space is a mix of sine and cosine functions:
# MAGIC
# MAGIC
# MAGIC $$PE_{(p, 2i)} = \sin\left(\frac{p}{10000^{2i/d}}\right)$$
# MAGIC
# MAGIC
# MAGIC
# MAGIC $$PE_{(p, 2i+1)} = \cos\left(\frac{p}{10000^{2i/d}}\right)$$
# MAGIC
# MAGIC
# MAGIC Here, \\(d\\) is the dimension of the word embedding.
# MAGIC
# MAGIC These functions were chosen because they can provide a unique encoding for each word position and these encodings can be easily learned and extrapolated for
# sequence lengths not seen during training.
# MAGIC
# MAGIC In the heatmap:
# MAGIC
# MAGIC - The x-axis represents the dimension of the embedding space. Every pair of dimensions \\((2i, 2i+1)\\) corresponds to a specific frequency of the sine and cosine functions.
# MAGIC
# MAGIC - The y-axis represents the position of a word in the sequence.
# MAGIC
# MAGIC - The color at each point in the heatmap represents the value of the positional encoding at that position and dimension. Typically, a warmer color (like red) represents a \
# higher value and a cooler color (like blue) represents a lower value.
# MAGIC
# MAGIC By visualizing the positional encodings in a heatmap, we can see how these values change across positions and dimensions, and get an intuition for how the Transformer model
# might use these values to understand the order of words in the sequence.

# COMMAND ----------

# Get positional encodings
max_seq_len = len(sentence.split())  # Maximum sequence length, 9
d_model = embedding_size  # Same as the size of the word embeddings, 16
positional_encodings = get_positional_encoding(max_seq_len, d_model)
"""
position: [[0]
 [1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]], shape: (9, 1)
div_term: [1.00000000e+00 3.16227766e-01 1.00000000e-01 3.16227766e-02
 1.00000000e-02 3.16227766e-03 1.00000000e-03 3.16227766e-04], shape: (8,)
positional_encoding: [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]], shape: (9, 16)
positional_encoding[:, 0::2]: [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
 [ 8.41470985e-01  3.10983593e-01  9.98334166e-02  3.16175064e-02
   9.99983333e-03  3.16227239e-03  9.99999833e-04  3.16227761e-04]
 [ 9.09297427e-01  5.91127117e-01  1.98669331e-01  6.32033979e-02
   1.99986667e-02  6.32451316e-03  1.99999867e-03  6.32455490e-04]
 [ 1.41120008e-01  8.12648897e-01  2.95520207e-01  9.47260913e-02
   2.99955002e-02  9.48669068e-03  2.99999550e-03  9.48683156e-04]
 [-7.56802495e-01  9.53580740e-01  3.89418342e-01  1.26154067e-01
   3.99893342e-02  1.26487733e-02  3.99998933e-03  1.26491073e-03]
 [-9.58924275e-01  9.99946517e-01  4.79425539e-01  1.57455898e-01
   4.99791693e-02  1.58107295e-02  4.99997917e-03  1.58113817e-03]
 [-2.79415498e-01  9.47148158e-01  5.64642473e-01  1.88600287e-01
   5.99640065e-02  1.89725276e-02  5.99996400e-03  1.89736546e-03]
 [ 6.56986599e-01  8.00421646e-01  6.44217687e-01  2.19556091e-01
   6.99428473e-02  2.21341359e-02  6.99994283e-03  2.21359255e-03]
 [ 9.89358247e-01  5.74317769e-01  7.17356091e-01  2.50292358e-01
   7.99146940e-02  2.52955229e-02  7.99991467e-03  2.52981943e-03]], shape: (9, 8)
positional_encoding[:, 1::2] : [[ 1.          1.          1.          1.          1.          1.
   1.          1.        ]
 [ 0.54030231  0.95041528  0.99500417  0.99950004  0.99995     0.999995
   0.9999995   0.99999995]
 [-0.41614684  0.80657841  0.98006658  0.99800067  0.99980001  0.99998
   0.999998    0.9999998 ]
 [-0.9899925   0.58275361  0.95533649  0.99550337  0.99955003  0.999955
   0.9999955   0.99999955]
 [-0.65364362  0.30113746  0.92106099  0.99201066  0.99920011  0.99992
   0.999992    0.9999992 ]
 [ 0.28366219 -0.01034232  0.87758256  0.98752602  0.99875026  0.999875
   0.9999875   0.99999875]
 [ 0.96017029 -0.32079646  0.82533561  0.98205394  0.99820054  0.99982001
   0.999982    0.9999982 ]
 [ 0.75390225 -0.59943739  0.76484219  0.97559988  0.997551    0.99975501
   0.9999755   0.99999755]
 [-0.14550003 -0.81863246  0.69670671  0.9681703   0.99680171  0.99968002
   0.999968    0.9999968 ]], shape: (9, 8)
torch.tensor(positional_encoding, dtype=torch.float): tensor([[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,
          0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          1.0000e+00],
        [ 8.4147e-01,  5.4030e-01,  3.1098e-01,  9.5042e-01,  9.9833e-02,
          9.9500e-01,  3.1618e-02,  9.9950e-01,  9.9998e-03,  9.9995e-01,
          3.1623e-03,  9.9999e-01,  1.0000e-03,  1.0000e+00,  3.1623e-04,
          1.0000e+00],
        [ 9.0930e-01, -4.1615e-01,  5.9113e-01,  8.0658e-01,  1.9867e-01,
          9.8007e-01,  6.3203e-02,  9.9800e-01,  1.9999e-02,  9.9980e-01,
          6.3245e-03,  9.9998e-01,  2.0000e-03,  1.0000e+00,  6.3246e-04,
          1.0000e+00],
        [ 1.4112e-01, -9.8999e-01,  8.1265e-01,  5.8275e-01,  2.9552e-01,
          9.5534e-01,  9.4726e-02,  9.9550e-01,  2.9996e-02,  9.9955e-01,
          9.4867e-03,  9.9995e-01,  3.0000e-03,  1.0000e+00,  9.4868e-04,
        [-7.5680e-01, -6.5364e-01,  9.5358e-01,  3.0114e-01,  3.8942e-01,
          9.2106e-01,  1.2615e-01,  9.9201e-01,  3.9989e-02,  9.9920e-01,
          1.2649e-02,  9.9992e-01,  4.0000e-03,  9.9999e-01,  1.2649e-03,
          1.0000e+00],
        [-9.5892e-01,  2.8366e-01,  9.9995e-01, -1.0342e-02,  4.7943e-01,
          8.7758e-01,  1.5746e-01,  9.8753e-01,  4.9979e-02,  9.9875e-01,
          1.5811e-02,  9.9988e-01,  5.0000e-03,  9.9999e-01,  1.5811e-03,
          1.0000e+00],
        [-2.7942e-01,  9.6017e-01,  9.4715e-01, -3.2080e-01,  5.6464e-01,
          8.2534e-01,  1.8860e-01,  9.8205e-01,  5.9964e-02,  9.9820e-01,
          1.8973e-02,  9.9982e-01,  6.0000e-03,  9.9998e-01,  1.8974e-03,
          1.0000e+00],
        [ 6.5699e-01,  7.5390e-01,  8.0042e-01, -5.9944e-01,  6.4422e-01,
          7.6484e-01,  2.1956e-01,  9.7560e-01,  6.9943e-02,  9.9755e-01,
          2.2134e-02,  9.9976e-01,  6.9999e-03,  9.9998e-01,  2.2136e-03,
          1.0000e+00],
        [ 9.8936e-01, -1.4550e-01,  5.7432e-01, -8.1863e-01,  7.1736e-01,
          6.9671e-01,  2.5029e-01,  9.6817e-01,  7.9915e-02,  9.9680e-01,
          2.5296e-02,  9.9968e-01,  7.9999e-03,  9.9997e-01,  2.5298e-03,
          1.0000e+00]]), shape: torch.Size([9, 16]) """


positional_encodings.shape  # torch.Size([9, 16])
word_embeddings.shape  # torch.Size([9, 16])

# Add word embeddings and positional encodings
final_embeddings = word_embeddings + positional_encodings
"""
tensor([[ 0.4696,  1.2848,  1.2779,  1.8044,  0.1650,  1.8639, -1.3978,  3.1976,
         -0.6877,  1.5311, -0.3997,  2.5169,  0.6127,  1.3905,  0.1623,  1.9743],
        [ 1.9389,  0.5714, -0.5461,  0.0085, -0.0834,  1.6880, -1.1635,  0.0609,
          0.2284,  0.8035,  1.3368, -1.1080, -0.4717,  0.0626,  1.3169,  1.3795],
        [-0.1735, -0.5383,  0.7555,  0.1125, -1.9051,  0.0615, -0.1289,  1.5912,
         -1.4648,  0.9081, -0.2167,  0.3034, -0.7650,  1.4881, -1.1317,  0.8642],
        [-1.6125, -0.6192,  1.0319,  0.3287,  1.6887,  1.6055, -0.5343,  0.3440,
          0.9020,  0.2229, -0.4924,  0.4389, -1.4621,  1.0365, -0.5425,  2.6658],
        [-1.2354, -0.3677,  0.4931,  2.2348,  0.6322,  0.9424,  0.7565, -0.1471,
          2.1235,  1.0827,  0.2784,  1.0845, -0.6659,  0.9493, -0.8194,  1.0192],
        [ 0.6698, -0.1127,  0.1836,  1.3903,  0.0958,  1.4113, -0.7755,  0.5756,
         -0.4613,  2.6544,  0.1583,  0.8714,  0.4691, -0.0608, -0.4323, -0.6315],
        [ 0.0054,  2.7346, -0.6590,  0.3141,  0.1781,  1.1468,  1.3135,  1.0909,
          1.7311,  1.5715, -0.9647,  0.4737,  0.5551, -0.6460,  1.6553,  1.7369],
        [ 0.8542, -1.2239,  2.4227,  0.4974, -0.5440,  0.8758, -0.3836,  0.3389,
         -0.1904,  1.4519, -0.2687,  2.5154,  0.6344,  0.0986, -0.8181,  2.7802],
        [-0.4522, -1.5623,  0.9864, -0.3546, -0.3502,  0.9216,  0.6888,  1.2231,
          0.2178, -0.6696,  1.4929,  2.6392,  0.2737,  1.7684,  1.6025,  0.7981]],
       grad_fn=<AddBackward0>) """

final_embeddings.shape  # torch.Size([9, 16])


print(final_embeddings)
"""
tensor([[ 0.4864, -0.2103, -2.0510,  0.0840,  0.6720,  1.2373, -1.2434,  1.7891,
         -0.9264,  0.9995, -0.3569,  1.1401,  1.3020,  1.5159,  0.0791,  0.5878],
        [-0.4321,  0.9168,  0.7751,  0.9667, -0.4041,  3.1663, -0.4536, -0.0871,
          0.8896,  1.0259,  0.7158,  2.6850,  0.7566,  0.4302, -1.9151,  1.0520],
        [ 0.4986, -1.3359,  1.6957,  1.7024, -1.5596,  2.0179,  0.0573,  1.1888,
          0.7119, -0.1685, -0.7062,  1.3399,  0.3312, -0.0721,  0.6834,  1.6510],
        [ 0.5503, -0.3466, -1.1335,  0.5843,  1.2802,  0.6891,  1.0213,  0.5967,
          0.8052,  0.7010, -1.1693,  1.0132,  2.1397,  0.7855,  1.0885,  2.4436],
        [-1.2769, -0.5281,  0.7301,  0.3387, -0.6644,  1.6687, -0.7273,  0.5593,
         -0.9942,  1.8367, -0.8638,  0.0914, -0.8234,  0.1648,  0.1998,  0.9707],
         -0.6063, -1.9212,  0.2266,  1.0384, -0.6887,  0.7737, -0.8265,  2.3337],
        [-0.4731,  1.9489,  0.2787, -1.9107,  0.4314, -0.4436, -0.9047,  2.2643,
          0.0667,  0.5758, -2.1327,  0.2057,  0.4958,  0.9402, -1.2666,  1.7354],
        [ 0.5128, -0.1348,  2.7128, -1.2578,  0.1834,  1.7824,  0.0215,  0.4955,
         -1.0845,  0.9885,  0.3081,  1.2553, -0.7795,  0.0859, -0.9598,  2.3965],
        [ 1.9994,  0.4741,  0.5960,  0.8354,  0.6924,  0.5946,  1.5498,  1.0064,
          0.1690,  2.4372,  0.2527, -0.9349, -0.5699, -0.4887,  0.2650,  0.9643]],
       grad_fn=<AddBackward0>) """

# COMMAND ----------

# MAGIC %md # Section 2: Building Our Own Decoder From Scratch
# MAGIC
# MAGIC Let's now build a decoder transformer. We'll build up the code from scratch and build a single layer transformer.

# COMMAND ----------

# Here we define the DecoderBlock, which is a single layer of the Transformer Decoder.


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout):
        super(DecoderBlock, self).__init__()

        # The first part of the __init__ function defines the hyperparameters for the DecoderBlock.
        # d_model: the dimension of the input vector.
        # num_heads: the number of heads in the multihead attention mechanism.
        # ff_hidden_dim: the dimension of the feed forward hidden layer.
        # dropout: the dropout rate.
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, ff_hidden_dim)
        self.linear2 = nn.Linear(ff_hidden_dim, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    # The forward method defines how the data flows through the network.
    # It takes two inputs: x, tgt_mask.
    # x: the input tensor.
    # tgt_mask: masks to prevent attention to certain positions.

    def forward(self, x, tgt_mask):
        print(
            f"x: {x}, x.shape: {x.shape}, tgt_mask: {tgt_mask}, tgt_mask.shape: {tgt_mask.shape}"
        )
        attn_output, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)
        print(
            f"attn_output: {attn_output}, , attn_output.shape: {attn_output.shape}, _: {_}, _.shape: {_.shape}"
        )
        print(f"self.dropout1(attn_output): {self.dropout1(attn_output)}")
        x = x + self.dropout1(attn_output)
        print(f"x after dropout: {x.shape}")
        x = self.norm1(x)
        print(f"x after norm1: {x.shape}")

        ff_output = self.linear2(F.relu(self.linear1(x)))
        print(f"ff_output: {ff_output}, ff_output shape: {ff_output.shape}")

        x = x + self.dropout2(ff_output)
        print(f"x after dropout 2: {x.shape}")

        x = self.norm2(x)
        print(f"x after norm2: {x.shape}")

        return x


# COMMAND ----------

# Next, we define the PositionalEncoding class, which applies a specific positional encoding to give the model
# information about the relative or absolute position of the tokens in the sequence.


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


# COMMAND ----------

# Finally, we define the full Transformer Decoder, which includes the initial embedding layer,
# a single Transformer Decoder block, and the final linear and softmax layers.


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_hidden_dim, dropout):
        super(TransformerDecoder, self).__init__()

        # The __init__ function defines the hyperparameters and layers of the TransformerDecoder.
        # vocab_size: the size of the vocabulary.
        # d_model, num_heads, ff_hidden_dim, dropout: hyperparameters for the Transformer decoder block.

        # Embedding layer: transforms the input words (given as indices) into dense vectors of dimension d_model.
        # Positional encoding: adds a vector to each input embedding that depends on its position in the sequence.
        # Transformer block: the Transformer decoder block defined earlier.
        # Linear layer: a linear transformation to the output dimension equal to the vocabulary size.
        # Softmax layer: transforms the output into a probability distribution over the vocabulary.

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_block = DecoderBlock(
            d_model, num_heads, ff_hidden_dim, dropout
        )
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    # The forward method of the TransformerDecoder defines how the data flows through the decoder.

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        tgt_mask = generate_square_subsequent_mask(x.size(0))
        x = self.transformer_block(x, tgt_mask)
        output = self.linear(x)
        output = self.softmax(output)
        return output


# COMMAND ----------

# MAGIC %md ### Why we need to mask our input for decoders

# COMMAND ----------


def generate_square_subsequent_mask(sz):
    """Generate a mask to prevent attention to future positions."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


mask = generate_square_subsequent_mask(sz=5)
"""
tensor([[0., -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0.]]) """


plt.figure(figsize=(5, 5))
seaborn.heatmap(mask, cmap="viridis", cbar=False, square=True)
plt.title("Mask for Transformer Decoder")
plt.show()

# COMMAND ----------

# MAGIC %md ### Let's make our first decoder

# COMMAND ----------

# Define the hyperparameters
vocab_size = 1000
d_model = 512
num_heads = 1
ff_hidden_dim = 2 * d_model
dropout = 0.1
num_layers = 10
context_length = 50
batch_size = 1
# Initialize the model
model = TransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_dim, dropout)

model
"""
TransformerDecoder(
  (embedding): Embedding(1000, 512)
  (pos_encoder): PositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (transformer_block): DecoderBlock(
    (self_attention): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
    )
    (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (dropout1): Dropout(p=0.1, inplace=False)
    (linear1): Linear(in_features=512, out_features=1024, bias=True)
    (linear2): Linear(in_features=1024, out_features=512, bias=True)
    (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (dropout2): Dropout(p=0.1, inplace=False)
  )
  (linear): Linear(in_features=512, out_features=1000, bias=True)
  (softmax): LogSoftmax(dim=-1)
) """

# Create a tensor representing a batch of 1 sequences of length 10
input_tensor = torch.randint(0, vocab_size, (context_length, batch_size))
"""
tensor([[306],
        [255],
        [948],
        [402],
        [262],
        [711],
        [565],
        [978],
        [345],
        [765],
        [470],
        [912],
        [998],
        [ 10],
        [774],
        [561],
        [289],
        [309],
        [814],
        [399],
        [189],
        [264],
        [195],
        [113],
        [827],
        [758],
        [780],
        [581],
        [394],
        [538],
        [180],
        [987],
        [660],
        [549],
        [985],
        [695],
        [631],
        [464],
        [513],
        [739],
        [198],
        [160],
        [602],
        [ 54],
        [413],
        [179],
        [844],
        [827],
        [384]]) """

input_tensor.shape  # torch.Size([50, 1])

# Forward pass through the model
output = model(input_tensor)
"""
x: tensor([[[ 1.2232, -0.8347, -2.6548,  ...,  2.1855, -0.3163,  0.9951]],

        [[ 0.0000,  1.0804, -0.0064,  ...,  0.5002,  0.1022,  2.4253]],

        [[-0.3915,  2.2227,  1.2315,  ...,  1.6442, -1.5591,  0.8986]],

        ...,

        [[ 0.3443, -0.3581, -0.0288,  ...,  2.2742,  1.7372,  0.4467]],

        [[-1.6178, -2.0346,  0.0000,  ..., -0.4284, -1.6105,  1.4083]],

        [[-0.4935,  0.9523, -0.6172,  ...,  0.8975, -1.0610,  2.7248]]],
       grad_fn=<MulBackward0>), x.shape: torch.Size([50, 1, 512]), tgt_mask: tensor([[0., -inf, -inf,  ..., -inf, -inf, -inf],
        [0., 0., -inf,  ..., -inf, -inf, -inf],
        [0., 0., 0.,  ..., -inf, -inf, -inf],
        ...,
        [0., 0., 0.,  ..., 0., -inf, -inf],
        [0., 0., 0.,  ..., 0., 0., -inf],
        [0., 0., 0.,  ..., 0., 0., 0.]]), tgt_mask.shape: torch.Size([50, 50])
attn_output: tensor([[[ 0.2275,  0.0303, -0.3356,  ...,  0.0639,  0.5058,  0.0971]],

        [[-0.2268, -0.4036,  0.2908,  ..., -0.4072,  0.1822,  0.2547]],

        [[ 0.2119, -0.0486, -0.0543,  ...,  0.0157,  0.4043, -0.0472]],

        ...,

        [[ 0.3094, -0.3398,  0.3588,  ..., -0.0124,  0.0820,  0.2248]],

        [[ 0.3084, -0.3311,  0.2609,  ..., -0.0100,  0.0306,  0.2047]],

        [[ 0.2088, -0.2527,  0.3210,  ...,  0.0120,  0.1510,  0.1398]]],
       grad_fn=<ViewBackward0>), , attn_output.shape: torch.Size([50, 1, 512]), _: tensor([[[1.1111, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
         [0.1510, 0.9601, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
         [0.8387, 0.1385, 0.1339,  ..., 0.0000, 0.0000, 0.0000],
         ...,
         [0.0140, 0.0309, 0.0071,  ..., 0.0089, 0.0000, 0.0000],
         [0.0102, 0.0150, 0.0185,  ..., 0.0181, 0.0180, 0.0000],
         [0.0310, 0.0000, 0.0519,  ..., 0.0095, 0.0400, 0.0247]]],
       grad_fn=<MeanBackward1>), _.shape: torch.Size([1, 50, 50])
self.dropout1(attn_output): tensor([[[ 0.2527,  0.0337, -0.3729,  ...,  0.0710,  0.5620,  0.1079]],

        [[-0.2520, -0.4485,  0.3231,  ..., -0.4524,  0.2024,  0.2830]],

        [[ 0.0000, -0.0540, -0.0603,  ...,  0.0175,  0.4492, -0.0524]],

        ...,

        [[ 0.3437, -0.3776,  0.3987,  ..., -0.0138,  0.0000,  0.2498]],

        [[ 0.0000, -0.3679,  0.2899,  ..., -0.0111,  0.0000,  0.2274]],

        [[ 0.2320, -0.2808,  0.3567,  ...,  0.0134,  0.1677,  0.1553]]],
       grad_fn=<MulBackward0>)
x after dropout: torch.Size([50, 1, 512])
x after norm1: torch.Size([50, 1, 512])
          -2.0785e-01, -2.3833e-01]],

        [[-2.0578e-02, -2.4346e-01,  1.7320e-01,  ..., -2.8718e-01,
           1.6908e-02,  1.0969e-01]],

        [[ 2.4662e-01,  4.0952e-01, -8.4569e-02,  ...,  1.3448e-01,
           1.4734e-02, -1.0322e-01]],

        ...,

        [[-3.5687e-02,  9.1817e-02,  3.0782e-02,  ...,  2.3952e-01,
          -1.9619e-01,  9.4854e-02]],

        [[-2.5924e-03, -2.3331e-01,  1.1474e-01,  ..., -9.7216e-02,
          -1.7706e-01, -1.5776e-01]],

        [[-1.4066e-01, -4.3738e-01, -4.8835e-05,  ..., -1.0988e-01,
          -6.2455e-01, -1.6553e-01]]], grad_fn=<ViewBackward0>), ff_output shape: torch.Size([50, 1, 512])
x after dropout 2: torch.Size([50, 1, 512])
x after norm2: torch.Size([50, 1, 512]) """

# The output is a tensor of shape (sequence_length, batch_size, vocab_size)
print(output.shape)  # Should print torch.Size([context_length, batch_size, vocab_size])
# torch.Size([50, 1, 1000])

# To get the predicted word indices, we can use the `argmax` function
predicted_indices = output.argmax(dim=-1)
"""
tensor([[281],
        [656],
        [246],
        [962],
        [418],
        [643],
        [  7],
        [ 74],
        [412],
        [ 20],
        [288],
        [501],
        [418],
        [124],
        [389],
        [370],
        [525],
        [266],
        [236],
        [346],
        [217],
        [431],
        [683],
        [347],
        [678],
        [658],
        [429],
        [346],
        [650],
        [615],
        [187],
        [424],
        [630],
        [842],
        [837],
        [413],
        [881],
        [351],
        [392],
        [633],
        [598],
        [805],
        [639],
        [127],
        [286],
        [847],
        [467],
        [678],
        [952]]) """

predicted_indices.shape  # torch.Size([50, 1])

# Now `predicted_indices` is a tensor of shape (sequence_length, batch_size) containing the predicted word indices
print(predicted_indices.shape)  # Should print torch.Size([context_length, batch_size])
# torch.Size([50, 1])


# COMMAND ----------


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")
# The model has 3,127,784 trainable parameters

# COMMAND ----------

# MAGIC %md ### Looking at the output

# COMMAND ----------

output
"""
tensor([[[-7.5383, -7.5449, -6.9099,  ..., -6.1569, -6.7795, -8.0481]],

        [[-7.1358, -6.9447, -6.9447,  ..., -6.1389, -7.4646, -6.9987]],

        [[-6.7621, -7.2041, -7.4765,  ..., -6.8076, -6.3738, -6.8126]],

        ...,

        [[-6.4850, -6.7661, -6.3359,  ..., -7.5549, -7.5813, -7.1425]],

        [[-7.0800, -7.5423, -6.9650,  ..., -6.7205, -7.1733, -8.0703]],

        [[-7.6994, -6.8403, -8.1435,  ..., -7.4831, -7.5201, -7.5780]]],
       grad_fn=<LogSoftmaxBackward0>) """

output.shape  # 50, 1, 1000

output[0, 0, :].shape

# Convert the log probabilities to probabilities
distribution = torch.exp(output[0, 0, :])
"""
tensor([0.0005, 0.0005, 0.0010, 0.0005, 0.0010, 0.0009, 0.0008, 0.0010, 0.0017,
        0.0004, 0.0016, 0.0010, 0.0017, 0.0010, 0.0006, 0.0035, 0.0022, 0.0010,
        0.0012, 0.0031, 0.0009, 0.0012, 0.0014, 0.0008, 0.0005, 0.0011, 0.0005,
        0.0005, 0.0007, 0.0007, 0.0006, 0.0013, 0.0005, 0.0021, 0.0012, 0.0025,
        0.0018, 0.0012, 0.0010, 0.0004, 0.0017, 0.0002, 0.0012, 0.0023, 0.0014,
        0.0005, 0.0011, 0.0014, 0.0005, 0.0005, 0.0010, 0.0005, 0.0011, 0.0008,
        0.0012, 0.0005, 0.0006, 0.0014, 0.0008, 0.0005, 0.0004, 0.0013, 0.0019,
        0.0006, 0.0007, 0.0003, 0.0005, 0.0016, 0.0006, 0.0009, 0.0005, 0.0013,
        0.0004, 0.0007, 0.0004, 0.0008, 0.0010, 0.0009, 0.0011, 0.0008, 0.0005,
        0.0007, 0.0004, 0.0016, 0.0010, 0.0008, 0.0016, 0.0007, 0.0008, 0.0006,
        0.0024, 0.0008, 0.0011, 0.0014, 0.0006, 0.0008, 0.0012, 0.0008, 0.0015,
        0.0007, 0.0008, 0.0015, 0.0016, 0.0004, 0.0008, 0.0009, 0.0005, 0.0010,
        0.0019, 0.0018, 0.0011, 0.0018, 0.0004, 0.0005, 0.0004, 0.0016, 0.0008,
        0.0006, 0.0007, 0.0023, 0.0004, 0.0008, 0.0006, 0.0007, 0.0017, 0.0006,
        0.0003, 0.0016, 0.0012, 0.0009, 0.0010, 0.0015, 0.0007, 0.0011, 0.0004,
        0.0003, 0.0004, 0.0022, 0.0010, 0.0017, 0.0016, 0.0008, 0.0006, 0.0008,
        0.0009, 0.0004, 0.0015, 0.0014, 0.0009, 0.0004, 0.0007, 0.0015, 0.0016,
        0.0018, 0.0005, 0.0012, 0.0003, 0.0004, 0.0019, 0.0014, 0.0008, 0.0008,
        0.0008, 0.0007, 0.0002, 0.0008, 0.0004, 0.0009, 0.0011, 0.0018, 0.0020,
        0.0005, 0.0011, 0.0006, 0.0013, 0.0019, 0.0006, 0.0014, 0.0008, 0.0018,
        0.0008, 0.0020, 0.0009, 0.0007, 0.0006, 0.0004, 0.0005, 0.0005, 0.0011,
        0.0009, 0.0016, 0.0014, 0.0003, 0.0010, 0.0010, 0.0007, 0.0008, 0.0015,
        0.0005, 0.0019, 0.0006, 0.0006, 0.0009, 0.0010, 0.0004, 0.0006, 0.0009,
        0.0006, 0.0005, 0.0022, 0.0013, 0.0005, 0.0014, 0.0004, 0.0010, 0.0005,
        0.0004, 0.0014, 0.0034, 0.0017, 0.0004, 0.0007, 0.0006, 0.0012, 0.0017,
        0.0010, 0.0004, 0.0008, 0.0010, 0.0006, 0.0018, 0.0011, 0.0006, 0.0005,
        0.0014, 0.0004, 0.0007, 0.0009, 0.0017, 0.0006, 0.0003, 0.0010, 0.0006,
        0.0013, 0.0002, 0.0016, 0.0010, 0.0008, 0.0003, 0.0025, 0.0003, 0.0015,
        0.0019, 0.0007, 0.0007, 0.0007, 0.0008, 0.0006, 0.0002, 0.0017, 0.0012,
        0.0008, 0.0022, 0.0005, 0.0021, 0.0008, 0.0009, 0.0004, 0.0007, 0.0008,
        0.0039, 0.0009, 0.0011, 0.0007, 0.0004, 0.0003, 0.0007, 0.0007, 0.0007,
        0.0010, 0.0028, 0.0044, 0.0016, 0.0008, 0.0004, 0.0012, 0.0004, 0.0014,
        0.0011, 0.0017, 0.0011, 0.0007, 0.0004, 0.0009, 0.0009, 0.0004, 0.0008,
        0.0006, 0.0013, 0.0004, 0.0017, 0.0006, 0.0011, 0.0007, 0.0012, 0.0009,
        0.0006, 0.0024, 0.0010, 0.0006, 0.0009, 0.0016, 0.0003, 0.0009, 0.0015,
        0.0018, 0.0015, 0.0018, 0.0003, 0.0009, 0.0005, 0.0008, 0.0017, 0.0008,
        0.0004, 0.0012, 0.0018, 0.0011, 0.0013, 0.0006, 0.0006, 0.0009, 0.0005,
        0.0017, 0.0010, 0.0008, 0.0018, 0.0009, 0.0009, 0.0011, 0.0013, 0.0012,
        0.0010, 0.0007, 0.0029, 0.0005, 0.0009, 0.0014, 0.0007, 0.0003, 0.0005,
        0.0020, 0.0008, 0.0005, 0.0006, 0.0009, 0.0003, 0.0004, 0.0005, 0.0012,
        0.0010, 0.0005, 0.0004, 0.0012, 0.0003, 0.0012, 0.0004, 0.0007, 0.0008,
        0.0009, 0.0008, 0.0009, 0.0002, 0.0006, 0.0004, 0.0006, 0.0006, 0.0041,
        0.0020, 0.0008, 0.0022, 0.0007, 0.0004, 0.0010, 0.0005, 0.0011, 0.0010,
        0.0011, 0.0007, 0.0023, 0.0006, 0.0012, 0.0006, 0.0007, 0.0004, 0.0005,
        0.0006, 0.0007, 0.0010, 0.0005, 0.0008, 0.0006, 0.0029, 0.0009, 0.0008,
        0.0009, 0.0016, 0.0010, 0.0005, 0.0006, 0.0006, 0.0009, 0.0017, 0.0008,
        0.0007, 0.0005, 0.0013, 0.0008, 0.0015, 0.0008, 0.0005, 0.0022, 0.0020,
        0.0006, 0.0013, 0.0007, 0.0015, 0.0034, 0.0011, 0.0008, 0.0006, 0.0009,
        0.0012, 0.0013, 0.0004, 0.0012, 0.0008, 0.0009, 0.0005, 0.0021, 0.0012,
        0.0013, 0.0008, 0.0008, 0.0008, 0.0004, 0.0005, 0.0006, 0.0005, 0.0005,
        0.0013, 0.0012, 0.0012, 0.0015, 0.0018, 0.0012, 0.0011, 0.0008, 0.0005,
        0.0009, 0.0004, 0.0014, 0.0004, 0.0011, 0.0009, 0.0018, 0.0012, 0.0009,
        0.0004, 0.0009, 0.0005, 0.0003, 0.0018, 0.0006, 0.0008, 0.0012, 0.0008,
        0.0006, 0.0008, 0.0005, 0.0025, 0.0003, 0.0008, 0.0009, 0.0021, 0.0006,
        0.0014, 0.0012, 0.0004, 0.0004, 0.0008, 0.0013, 0.0004, 0.0007, 0.0017,
        0.0004, 0.0007, 0.0022, 0.0007, 0.0014, 0.0005, 0.0020, 0.0009, 0.0007,
        0.0015, 0.0007, 0.0029, 0.0006, 0.0004, 0.0009, 0.0006, 0.0003, 0.0010,
        0.0003, 0.0029, 0.0009, 0.0005, 0.0003, 0.0009, 0.0005, 0.0012, 0.0005,
        0.0021, 0.0026, 0.0003, 0.0006, 0.0009, 0.0010, 0.0013, 0.0011, 0.0004,
        0.0003, 0.0004, 0.0009, 0.0008, 0.0008, 0.0015, 0.0003, 0.0029, 0.0007,
        0.0011, 0.0006, 0.0012, 0.0011, 0.0009, 0.0005, 0.0007, 0.0008, 0.0007,
        0.0004, 0.0014, 0.0004, 0.0013, 0.0003, 0.0008, 0.0009, 0.0008, 0.0005,
        0.0007, 0.0006, 0.0017, 0.0005, 0.0015, 0.0020, 0.0010, 0.0005, 0.0004,
        0.0013, 0.0017, 0.0023, 0.0014, 0.0009, 0.0010, 0.0011, 0.0004, 0.0012,
        0.0004, 0.0007, 0.0012, 0.0015, 0.0010, 0.0006, 0.0003, 0.0013, 0.0006,
        0.0009, 0.0006, 0.0014, 0.0004, 0.0006, 0.0014, 0.0010, 0.0006, 0.0013,
        0.0013, 0.0002, 0.0005, 0.0005, 0.0015, 0.0006, 0.0008, 0.0033, 0.0006,
        0.0006, 0.0015, 0.0008, 0.0004, 0.0005, 0.0012, 0.0006, 0.0010, 0.0005,
        0.0005, 0.0007, 0.0008, 0.0016, 0.0011, 0.0009, 0.0033, 0.0008, 0.0007,
        0.0009, 0.0008, 0.0005, 0.0010, 0.0002, 0.0013, 0.0018, 0.0008, 0.0016,
        0.0013, 0.0011, 0.0007, 0.0008, 0.0007, 0.0014, 0.0014, 0.0009, 0.0007,
        0.0005, 0.0004, 0.0021, 0.0022, 0.0017, 0.0031, 0.0024, 0.0013, 0.0008,
        0.0012, 0.0009, 0.0016, 0.0005, 0.0009, 0.0016, 0.0003, 0.0005, 0.0003,
        0.0003, 0.0004, 0.0004, 0.0017, 0.0007, 0.0011, 0.0010, 0.0005, 0.0010,
        0.0006, 0.0012, 0.0007, 0.0010, 0.0012, 0.0009, 0.0005, 0.0005, 0.0017,
        0.0006, 0.0008, 0.0004, 0.0012, 0.0011, 0.0008, 0.0027, 0.0013, 0.0007,
        0.0026, 0.0004, 0.0008, 0.0016, 0.0011, 0.0005, 0.0023, 0.0009, 0.0021,
        0.0006, 0.0028, 0.0009, 0.0010, 0.0019, 0.0010, 0.0006, 0.0007, 0.0010,
        0.0006, 0.0012, 0.0013, 0.0007, 0.0014, 0.0008, 0.0006, 0.0006, 0.0015,
        0.0018, 0.0008, 0.0007, 0.0007, 0.0014, 0.0006, 0.0009, 0.0004, 0.0010,
        0.0006, 0.0009, 0.0008, 0.0004, 0.0018, 0.0014, 0.0015, 0.0004, 0.0013,
        0.0008, 0.0004, 0.0008, 0.0015, 0.0011, 0.0033, 0.0005, 0.0012, 0.0003,
        0.0011, 0.0006, 0.0021, 0.0033, 0.0011, 0.0005, 0.0007, 0.0028, 0.0010,
        0.0004, 0.0005, 0.0008, 0.0006, 0.0004, 0.0006, 0.0003, 0.0007, 0.0021,
        0.0019, 0.0008, 0.0005, 0.0007, 0.0012, 0.0008, 0.0015, 0.0012, 0.0010,
        0.0009, 0.0007, 0.0007, 0.0007, 0.0006, 0.0010, 0.0007, 0.0008, 0.0016,
        0.0008, 0.0008, 0.0010, 0.0009, 0.0012, 0.0019, 0.0016, 0.0006, 0.0007,
        0.0010, 0.0005, 0.0012, 0.0004, 0.0005, 0.0012, 0.0004, 0.0005, 0.0015,
        0.0005, 0.0008, 0.0009, 0.0020, 0.0017, 0.0013, 0.0003, 0.0002, 0.0007,
        0.0006, 0.0007, 0.0004, 0.0008, 0.0007, 0.0007, 0.0008, 0.0003, 0.0007,
        0.0014, 0.0006, 0.0006, 0.0016, 0.0016, 0.0008, 0.0006, 0.0019, 0.0006,
        0.0009, 0.0025, 0.0009, 0.0004, 0.0010, 0.0009, 0.0007, 0.0007, 0.0006,
        0.0007, 0.0008, 0.0008, 0.0006, 0.0008, 0.0006, 0.0018, 0.0010, 0.0014,
        0.0024, 0.0015, 0.0014, 0.0002, 0.0004, 0.0007, 0.0008, 0.0010, 0.0010,
        0.0006, 0.0005, 0.0008, 0.0014, 0.0007, 0.0008, 0.0012, 0.0013, 0.0004,
        0.0006, 0.0020, 0.0006, 0.0010, 0.0010, 0.0012, 0.0021, 0.0019, 0.0006,
        0.0006, 0.0009, 0.0009, 0.0011, 0.0010, 0.0004, 0.0015, 0.0013, 0.0006,
        0.0008, 0.0007, 0.0005, 0.0005, 0.0008, 0.0008, 0.0006, 0.0011, 0.0011,
        0.0005, 0.0007, 0.0003, 0.0007, 0.0008, 0.0010, 0.0006, 0.0007, 0.0008,
        0.0004, 0.0009, 0.0006, 0.0009, 0.0009, 0.0013, 0.0011, 0.0012, 0.0009,
        0.0011, 0.0006, 0.0020, 0.0008, 0.0011, 0.0011, 0.0009, 0.0006, 0.0007,
        0.0013, 0.0005, 0.0008, 0.0015, 0.0011, 0.0021, 0.0011, 0.0004, 0.0007,
        0.0008, 0.0007, 0.0016, 0.0012, 0.0014, 0.0008, 0.0009, 0.0013, 0.0011,
        0.0006, 0.0011, 0.0009, 0.0004, 0.0011, 0.0009, 0.0008, 0.0009, 0.0015,
        0.0009, 0.0007, 0.0009, 0.0023, 0.0016, 0.0013, 0.0014, 0.0004, 0.0023,
        0.0005, 0.0039, 0.0007, 0.0006, 0.0018, 0.0011, 0.0003, 0.0003, 0.0003,
        0.0009, 0.0011, 0.0010, 0.0020, 0.0009, 0.0007, 0.0016, 0.0012, 0.0004,
        0.0009, 0.0010, 0.0017, 0.0006, 0.0021, 0.0008, 0.0013, 0.0022, 0.0004,
        0.0010, 0.0016, 0.0005, 0.0003, 0.0009, 0.0012, 0.0007, 0.0006, 0.0010,
        0.0005, 0.0009, 0.0006, 0.0012, 0.0006, 0.0009, 0.0013, 0.0021, 0.0011,
        0.0003], grad_fn=<ExpBackward0>) """


# Convert the output tensor to numpy array
distribution = distribution.detach().numpy()

# Now plot the distribution
plt.figure(figsize=(12, 6))
plt.bar(np.arange(vocab_size), distribution)
plt.xlabel("Word Index")
plt.ylabel("Probability")
plt.title("Output Distribution over Vocabulary")
plt.show()

# COMMAND ----------

# MAGIC %md # Section 3: Multi-layer Decoder
# MAGIC
# MAGIC Let's allow for multiple layers in our decoder so we can form models like GPT

# COMMAND ----------


class MultiLayerTransformerDecoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_heads, ff_hidden_dim, dropout, num_layers
    ):
        super(MultiLayerTransformerDecoder, self).__init__()

        # The __init__ function now also takes a `num_layers` argument, which specifies the number of decoder blocks.

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, num_heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    # The forward method has been updated to pass the input through each transformer block in sequence.

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for transformer_block in self.transformer_blocks:
            tgt_mask = generate_square_subsequent_mask(x.size(0))
            x = transformer_block(x, tgt_mask)
        output = self.linear(x)
        output = self.softmax(output)
        return output


# COMMAND ----------

# Define the hyperparameters
vocab_size = 10000
d_model = 2048
num_heads = 1
ff_hidden_dim = 4 * d_model
dropout = 0.1
num_layers = 10
context_length = 100
batch_size = 1

# Create our input to the model to process
input_tensor = torch.randint(0, vocab_size, (context_length, batch_size))

# Initialize the model with `num_layer` layers
model = MultiLayerTransformerDecoder(
    vocab_size, d_model, num_heads, ff_hidden_dim, dropout, num_layers
)
"""
  (embedding): Embedding(10000, 2048)
  (pos_encoder): PositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (transformer_blocks): ModuleList(
    (0-9): 10 x DecoderBlock(
      (self_attention): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=2048, out_features=2048, bias=True)
      )
      (norm1): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      (dropout1): Dropout(p=0.1, inplace=False)
      (linear1): Linear(in_features=2048, out_features=8192, bias=True)
      (linear2): Linear(in_features=8192, out_features=2048, bias=True)
      (norm2): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      (dropout2): Dropout(p=0.1, inplace=False)
    )
  )
  (linear): Linear(in_features=2048, out_features=10000, bias=True)
  (softmax): LogSoftmax(dim=-1)
) """

# Print the number of trainable parameters
print(f"The model has {count_parameters(model):,} trainable parameters")
# The model has 544,552,720 trainable parameters

# Let's use the same input_tensor from the previous example
output = model(input_tensor)

# Convert the log probabilities to probabilities for the first sequence in the batch and the first position in the sequence
distribution = torch.exp(output[0, 0, :])

# Convert the output tensor to numpy array
distribution = distribution.detach().numpy()

# Now plot the distribution
plt.figure(figsize=(12, 6))
plt.bar(np.arange(vocab_size), distribution)
plt.xlabel("Word Index")
plt.ylabel("Probability")
plt.title("Output Distribution over Vocabulary")
plt.show()

# COMMAND ----------

model
"""
MultiLayerTransformerDecoder(
  (embedding): Embedding(10000, 2048)
  (pos_encoder): PositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (transformer_blocks): ModuleList(
    (0-9): 10 x DecoderBlock(
      (self_attention): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=2048, out_features=2048, bias=True)
      )
      (norm1): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      (dropout1): Dropout(p=0.1, inplace=False)
      (linear2): Linear(in_features=8192, out_features=2048, bias=True)
      (norm2): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      (dropout2): Dropout(p=0.1, inplace=False)
    )
  )
  (linear): Linear(in_features=2048, out_features=10000, bias=True)
  (softmax): LogSoftmax(dim=-1)
)
 """


# COMMAND ----------

# MAGIC %md # Section 4: Adding real vocabulary to our model
# MAGIC
# MAGIC Rather than just using a random integer, let's add in a small vocabulary of real words and let our model speak!

# COMMAND ----------

# Define the hyperparameters
d_model = 100
num_heads = 1
ff_hidden_dim = 4 * d_model
dropout = 0.1
num_layers = 4
context_length = 5
batch_size = 1
# Define the vocabulary
vocab = [
    "of",
    "in",
    "to",
    "for",
    "with",
    "on",
    "at",
    "from",
    "by",
    "about",
    "as",
    "into",
    "like",
    "through",
    "after",
    "over",
    "between",
    "out",
    "against",
    "during",
    "without",
    "before",
    "under",
    "around",
    "among",
]
vocab_size = len(vocab)  # 25

# Create a dictionary that maps words to indices
word2id = {word: id for id, word in enumerate(vocab)}
"""
{'of': 0, 'in': 1, 'to': 2, 'for': 3, 'with': 4, 'on': 5, 'at': 6, 'from': 7, 'by': 8, 'about': 9, 'as': 10, 'into': 11, 'like': 12, 'through': 13, 'after': 14, 'over': 15, 'between': 16, 'out': 17, 'against': 18, 'during': 19, 'without': 20, 'before': 21, 'under': 22, 'around': 23, 'among': 24} """

# Create a dictionary that maps indices to words
id2word = {id: word for id, word in enumerate(vocab)}
"""
{0: 'of', 1: 'in', 2: 'to', 3: 'for', 4: 'with', 5: 'on', 6: 'at', 7: 'from', 8: 'by', 9: 'about', 10: 'as', 11: 'into', 12: 'like', 13: 'through', 14: 'after', 15: 'over', 16: 'between', 17: 'out', 18: 'against', 19: 'during', 20: 'without', 21: 'before', 22: 'under', 23: 'around', 24: 'among'} """

# Initialize the model
model = MultiLayerTransformerDecoder(
    vocab_size, d_model, num_heads, ff_hidden_dim, dropout, num_layers
)

# Create a tensor representing a single sequence of variable length
# Here we randomly select words from our vocabulary
sequence = ["of", "in", "to", "for", "with", "on", "at"][:context_length]
input_tensor = torch.tensor([[word2id[word] for word in sequence]])

# Generate a sequence of words
generated_words = []
for i in range(10):  # Generate 10 words
    output = model(input_tensor)
    predicted_index = output.argmax(dim=-1)[0, -1]  # Take the last word in the sequence
    predicted_word = id2word[predicted_index.item()]
    print(predicted_word, end=" ")
    generated_words.append(predicted_word)
    input_tensor = torch.cat(
        [input_tensor, predicted_index.unsqueeze(0).unsqueeze(0)], dim=-1
    )  # Append the predicted word to the input
    time.sleep(0.75)  # Pause for 1 second

generated_words
# ['before', 'as', 'about', 'over', 'against', 'into', 'as', 'through', 'to', 'before']

# COMMAND ----------

# MAGIC %md # Section 5: Using a trained decoder and real-world vocabulary
# MAGIC
# MAGIC Training our model will take a long time, let's look at two trained versions of what we've been building, GPT and GPT-XL. These are both decoder models with only slight changes in sizes

# COMMAND ----------


# Load pre-trained models and tokenizers
tokenizer_small = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=".\\models")
model_small = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=".\\models")

# COMMAND ----------

# Define a prompt. This is the initial string of text that the model will use to start generating text.
prompt = (
    "This is a MOOC about large language models, I have only just started, but already"
)

# COMMAND ----------

# We use the tokenizer to convert the prompt into a format that the model can understand. In this case,
# it converts the string into a sequence of token IDs, which are numbers that represent each word or subword in the string.
inputs_small = tokenizer_small.encode(prompt, return_tensors="pt")

# Create an attention mask. This is a sequence of 1s and 0s where 1s indicate that the corresponding token should
# be attended to and 0s indicate that the token should be ignored. Here, all tokens should be attended to.
attention_mask_small = torch.ones(inputs_small.shape, dtype=torch.long)

# Get the ID of the special end-of-sequence (EOS) token from the tokenizer. This token indicates the end of a sequence.
pad_token_id_small = tokenizer_small.eos_token_id

# Print the initial prompt. The 'end' argument specifies what to print at the end (default is newline, but we want space).
# 'flush' argument ensures that the output is printed immediately.
print(prompt, end=" ", flush=True)

# We're going to generate 25 words
for _ in range(25):
    # Generate the next part of the sequence. 'do_sample=True' means to sample from the distribution of possible next tokens
    # rather than just taking the most likely next token. 'pad_token_id' argument is to tell the model what token to use if it
    # needs to pad the sequence to a certain length.
    outputs_small = model_small.generate(
        inputs_small,
        max_length=inputs_small.shape[-1] + 1,
        do_sample=True,
        pad_token_id=pad_token_id_small,
        attention_mask=attention_mask_small,
    )

    # The generated output is a sequence of token IDs, so we use the tokenizer to convert these back into words.
    generated_word = tokenizer_small.decode(outputs_small[0][-1])

    # Print the generated word, followed by a space. We use 'end' and 'flush' arguments as before.
    print(generated_word, end=" ", flush=True)

    # Append the generated token to the input sequence for the next round of generation. We have to add extra dimensions
    # to the tensor to match the shape of the input tensor (which is 2D: batch size x sequence length).
    inputs_small = torch.cat(
        [inputs_small, outputs_small[0][-1].unsqueeze(0).unsqueeze(0)], dim=-1
    )

    # Extend the attention mask for the new token. Like before, it should be attended to, so we add a 1.
    attention_mask_small = torch.cat(
        [attention_mask_small, torch.ones((1, 1), dtype=torch.long)], dim=-1
    )

    # We pause for 0.7 seconds to make the generation more readable.
    time.sleep(0.7)

# Finally, print a newline and a completion message.
print("\nGPT-2 Small completed.")

# COMMAND ----------

tokenizer_large = GPT2Tokenizer.from_pretrained("gpt2-XL", cache_dir=".\\models")
model_large = GPT2LMHeadModel.from_pretrained("gpt2-XL", cache_dir=".\\models")

# COMMAND ----------

# Generate text with GPT-2 XL
inputs_large = tokenizer_large.encode(prompt, return_tensors="pt")

# Add in the attention mask and pad token id
attention_mask_large = torch.ones(
    inputs_large.shape, dtype=torch.long
)  # Creating a mask of ones with the same shape as inputs
pad_token_id_large = (
    tokenizer_large.eos_token_id
)  # Get the eos_token_id from the tokenizer

print(prompt, end=" ", flush=True)
for _ in range(25):  # Generate 25 words
    outputs_large = model_large.generate(
        inputs_large,
        max_length=inputs_large.shape[-1] + 1,
        do_sample=True,
        pad_token_id=pad_token_id_large,
        attention_mask=attention_mask_large,
    )
    generated_word = tokenizer_large.decode(outputs_large[0][-1])
    print(generated_word, end=" ", flush=True)
    inputs_large = torch.cat(
        [inputs_large, outputs_large[0][-1].unsqueeze(0).unsqueeze(0)], dim=-1
    )
    attention_mask_large = torch.cat(
        [attention_mask_large, torch.ones((1, 1), dtype=torch.long)], dim=-1
    )
    time.sleep(0.7)
print("\nGPT-2 XL completed.")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> |
# <a href="https://help.databricks.com/">Support</a>
