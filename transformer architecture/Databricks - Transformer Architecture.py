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
# MAGIC This lesson introduces the underlying structure of transformers from token management to the layers in a decoder, to comparing smaller and larger models. We will build up all of the steps needed to create our foundation model before training. You will see how the layers are constructed, and how the next word is chosen.
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

# COMMAND ----------

# MAGIC %md # Section 1: Encoding Natural Language - Word Embedding and Positional Encoding
# MAGIC
# MAGIC In this section we'll look at how to take a natural language input and convert it to the form we'll need for our transformer.

# COMMAND ----------

# Define a sentence and a simple word2id mapping
sentence = "The quick brown fox jumps over the lazy dog"
word2id = {word: i for i, word in enumerate(set(sentence.split()))}
print(word2id) # {'dog': 0, 'The': 1, 'over': 2, 'the': 3, 'fox': 4, 'quick': 5, 'jumps': 6, 'brown': 7, 'lazy': 8}

# Convert text to indices
input_ids = torch.tensor([word2id[word] for word in sentence.split()])
print(input_ids) # tensor([1, 5, 7, 4, 6, 2, 3, 8, 0])

# Define a simple word embedding function
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
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    positional_encoding = np.zeros((max_seq_len, d_model))
    positional_encoding[:, 0::2] = np.sin(position * div_term)
    positional_encoding[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(positional_encoding, dtype=torch.float)


# COMMAND ----------

# Function to plot heatmap
# ------------------------
def plot_heatmap(data, title):
    plt.figure(figsize=(5,5))
    seaborn.heatmap(data, cmap="cool",vmin=-1, vmax=1)
    plt.ylabel("Word/token")
    plt.xlabel("Positional Encoding Vector")
    plt.title(title)
    plt.show()

# Generate and plot positional encoding
# -------------------------------------
# Get positional encodings
max_seq_len = len(sentence.split())  # Maximum sequence length
d_model = embedding_size  # Same as the size of the word embeddings
positional_encodings = get_positional_encoding(max_seq_len, d_model)
plot_heatmap(positional_encodings, "Positional Encoding")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpreting the Positional Encoding Map
# MAGIC In the Transformer model, positional encoding is used to give the model some information about the relative positions of the words in the sequence since the Transformer does not have any inherent sense of order of the input sequence.
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
# MAGIC These functions were chosen because they can provide a unique encoding for each word position and these encodings can be easily learned and extrapolated for sequence lengths not seen during training.
# MAGIC
# MAGIC In the heatmap:
# MAGIC
# MAGIC - The x-axis represents the dimension of the embedding space. Every pair of dimensions \\((2i, 2i+1)\\) corresponds to a specific frequency of the sine and cosine functions.
# MAGIC
# MAGIC - The y-axis represents the position of a word in the sequence.
# MAGIC
# MAGIC - The color at each point in the heatmap represents the value of the positional encoding at that position and dimension. Typically, a warmer color (like red) represents a higher value and a cooler color (like blue) represents a lower value.
# MAGIC
# MAGIC By visualizing the positional encodings in a heatmap, we can see how these values change across positions and dimensions, and get an intuition for how the Transformer model might use these values to understand the order of words in the sequence.

# COMMAND ----------

# Get positional encodings
max_seq_len = len(sentence.split())  # Maximum sequence length
d_model = embedding_size  # Same as the size of the word embeddings
positional_encodings = get_positional_encoding(max_seq_len, d_model)

# Add word embeddings and positional encodings
final_embeddings = word_embeddings + positional_encodings

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

        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
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

    def forward(self, x,tgt_mask):
        attn_output, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
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
        self.transformer_block = DecoderBlock(d_model, num_heads, ff_hidden_dim, dropout)
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    # The forward method of the TransformerDecoder defines how the data flows through the decoder.

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        tgt_mask = generate_square_subsequent_mask(x.size(0))
        x = self.transformer_block(x,tgt_mask)
        output = self.linear(x)
        output = self.softmax(output)
        return output

# COMMAND ----------

# MAGIC %md ### Why we need to mask our input for decoders

# COMMAND ----------

def generate_square_subsequent_mask(sz):
    """Generate a mask to prevent attention to future positions."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

mask = generate_square_subsequent_mask(sz=5)

plt.figure(figsize=(5,5))
seaborn.heatmap(mask, cmap="viridis", cbar=False, square=True)
plt.title("Mask for Transformer Decoder")
plt.show()

# COMMAND ----------

# MAGIC %md ### Let's make our first decoder

# COMMAND ----------

# Define the hyperparameters
vocab_size     = 1000
d_model        = 512
num_heads      = 1
ff_hidden_dim  = 2*d_model
dropout        = 0.1
num_layers     = 10
context_length = 50
batch_size     = 1
# Initialize the model
model = TransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_dim, dropout)

# Create a tensor representing a batch of 1 sequences of length 10
input_tensor = torch.randint(0, vocab_size, (context_length, batch_size))

# Forward pass through the model
output = model(input_tensor)

# The output is a tensor of shape (sequence_length, batch_size, vocab_size)
print(output.shape)  # Should print torch.Size([context_length, batch_size, vocab_size])
# torch.Size([50, 1, 1000])

# To get the predicted word indices, we can use the `argmax` function
predicted_indices = output.argmax(dim=-1)

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

# Convert the log probabilities to probabilities
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

# MAGIC %md # Section 3: Multi-layer Decoder
# MAGIC
# MAGIC Let's allow for multiple layers in our decoder so we can form models like GPT

# COMMAND ----------

class MultiLayerTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_hidden_dim, dropout, num_layers):
        super(MultiLayerTransformerDecoder, self).__init__()

# The __init__ function now also takes a `num_layers` argument, which specifies the number of decoder blocks.

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

# The forward method has been updated to pass the input through each transformer block in sequence.

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for transformer_block in self.transformer_blocks:
            tgt_mask = generate_square_subsequent_mask(x.size(0))
            x = transformer_block(x,tgt_mask)
        output = self.linear(x)
        output = self.softmax(output)
        return output


# COMMAND ----------

# Define the hyperparameters
vocab_size     = 10000
d_model        = 2048
num_heads      = 1
ff_hidden_dim  = 4*d_model
dropout        = 0.1
num_layers     = 10
context_length = 100
batch_size     = 1

# Create our input to the model to process
input_tensor = torch.randint(0, vocab_size, (context_length, batch_size))

# Initialize the model with `num_layer` layers
model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_dim, dropout, num_layers)

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
>>> model
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
) """


# COMMAND ----------

# MAGIC %md # Section 4: Adding real vocabulary to our model
# MAGIC
# MAGIC Rather than just using a random integer, let's add in a small vocabulary of real words and let our model speak!

# COMMAND ----------

# Define the hyperparameters
d_model        = 100
num_heads      = 1
ff_hidden_dim  = 4*d_model
dropout        = 0.1
num_layers     = 4
context_length = 5
batch_size     = 1
# Define the vocabulary
vocab = ["of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "as", "into", "like", "through", "after", "over", "between", "out", "against", "during", "without", "before", "under", "around", "among"]
vocab_size = len(vocab) # 25

# Create a dictionary that maps words to indices
word2id = {word: id for id, word in enumerate(vocab)}
"""
{'of': 0, 'in': 1, 'to': 2, 'for': 3, 'with': 4, 'on': 5, 'at': 6, 'from': 7, 'by': 8, 'about': 9, 'as': 10, 'into': 11, 'like': 12, 'through': 13, 'after': 14, 'over': 15, 'between': 16, 'out': 17, 'against': 18, 'during': 19, 'without': 20, 'before': 21, 'under': 22, 'around': 23, 'among': 24} """

# Create a dictionary that maps indices to words
id2word = {id: word for id, word in enumerate(vocab)}
"""
{0: 'of', 1: 'in', 2: 'to', 3: 'for', 4: 'with', 5: 'on', 6: 'at', 7: 'from', 8: 'by', 9: 'about', 10: 'as', 11: 'into', 12: 'like', 13: 'through', 14: 'after', 15: 'over', 16: 'between', 17: 'out', 18: 'against', 19: 'during', 20: 'without', 21: 'before', 22: 'under', 23: 'around', 24: 'among'} """

# Initialize the model
model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_dim, dropout, num_layers)

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
    input_tensor = torch.cat([input_tensor, predicted_index.unsqueeze(0).unsqueeze(0)], dim=-1)  # Append the predicted word to the input
    time.sleep(0.75)  # Pause for 1 second


# COMMAND ----------

# MAGIC %md # Section 5: Using a trained decoder and real-world vocabulary
# MAGIC
# MAGIC Training our model will take a long time, let's look at two trained versions of what we've been building, GPT and GPT-XL. These are both decoder models with only slight changes in sizes

# COMMAND ----------

# Import the necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained models and tokenizers
tokenizer_small = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=DA.paths.datasets+"/models")
model_small = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=DA.paths.datasets+"/models")

# COMMAND ----------

# Define a prompt. This is the initial string of text that the model will use to start generating text.
prompt = "This is a MOOC about large language models, I have only just started, but already"

# COMMAND ----------

# We use the tokenizer to convert the prompt into a format that the model can understand. In this case,
# it converts the string into a sequence of token IDs, which are numbers that represent each word or subword in the string.
inputs_small = tokenizer_small.encode(prompt, return_tensors='pt')

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
    outputs_small = model_small.generate(inputs_small, max_length=inputs_small.shape[-1]+1, do_sample=True, pad_token_id=pad_token_id_small,
                                         attention_mask=attention_mask_small)

    # The generated output is a sequence of token IDs, so we use the tokenizer to convert these back into words.
    generated_word = tokenizer_small.decode(outputs_small[0][-1])

    # Print the generated word, followed by a space. We use 'end' and 'flush' arguments as before.
    print(generated_word, end=' ', flush=True)

    # Append the generated token to the input sequence for the next round of generation. We have to add extra dimensions
    # to the tensor to match the shape of the input tensor (which is 2D: batch size x sequence length).
    inputs_small = torch.cat([inputs_small, outputs_small[0][-1].unsqueeze(0).unsqueeze(0)], dim=-1)

    # Extend the attention mask for the new token. Like before, it should be attended to, so we add a 1.
    attention_mask_small = torch.cat([attention_mask_small, torch.ones((1, 1), dtype=torch.long)], dim=-1)

    # We pause for 0.7 seconds to make the generation more readable.
    time.sleep(0.7)

# Finally, print a newline and a completion message.
print("\nGPT-2 Small completed.")

# COMMAND ----------

tokenizer_large = GPT2Tokenizer.from_pretrained("gpt2-XL", cache_dir=DA.paths.datasets+"/models")
model_large = GPT2LMHeadModel.from_pretrained("gpt2-XL", cache_dir=DA.paths.datasets+"/models")

# COMMAND ----------

# Generate text with GPT-2 XL
inputs_large = tokenizer_large.encode(prompt, return_tensors="pt")

# Add in the attention mask and pad token id
attention_mask_large = torch.ones(inputs_large.shape, dtype=torch.long)  # Creating a mask of ones with the same shape as inputs
pad_token_id_large = tokenizer_large.eos_token_id  # Get the eos_token_id from the tokenizer

print(prompt, end=" ", flush=True)
for _ in range(25):  # Generate 25 words
    outputs_large = model_large.generate(inputs_large, max_length=inputs_large.shape[-1]+1, do_sample=True, pad_token_id=pad_token_id_large,
                                         attention_mask=attention_mask_large)
    generated_word = tokenizer_large.decode(outputs_large[0][-1])
    print(generated_word, end=" ", flush=True)
    inputs_large = torch.cat([inputs_large, outputs_large[0][-1].unsqueeze(0).unsqueeze(0)], dim=-1)
    attention_mask_large = torch.cat([attention_mask_large, torch.ones((1, 1), dtype=torch.long)], dim=-1)
    time.sleep(0.7)
print("\nGPT-2 XL completed.")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
