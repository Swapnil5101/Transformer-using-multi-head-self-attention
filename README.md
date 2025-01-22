# cnn_Transformer for Audio Classification using multi-head self-attention
Download the Esc-50 Environmental sound classification dataset from web.
This projects uses multi-head self-attention mechanism (inspired from famous "Attention is all you need" research paper) to classify some different kinds of sounds.

This repository contains an implementation of a hybrid architecture combining convolutional layers and transformers for classification tasks. The cnn_Transformer model leverages the strengths of both convolutional neural networks (CNNs) and transformers to achieve effective feature extraction and classification.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The cnn_Transformer model combines convolutional layers for local feature extraction and Encoder-only Transformers for capturing global dependencies. This hybrid approach allows the model to effectively learn both local and global patterns in the input data, making it suitable for various classification tasks.

## Model Architecture

The cnn_Transformer model consists of the following components:

1. **Convolutional Layers**: A series of 1D convolutional layers are used to extract local features from the input data. Each convolutional layer is followed by batch normalization, LeakyReLU activation, and average pooling.

2. **Positional Encoding**: Positional encoding is added to the input sequence to incorporate positional information into the transformer layers.

3. **Transformer Blocks**: The transformer blocks consist of multi-head self-attention and feed-forward layers. The self-attention mechanism allows the model to capture dependencies between different positions in the input sequence.

4. **Classification Head**: The output of the transformer blocks is averaged globally, and then passed through an MLP head for classification. The MLP head consists of linear layers with batch normalization, LeakyReLU activation, and dropout.

## Installation

To use the cnn_Transformer model, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/sm1899/cnn_Transformer-classification.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Here's an example of how to use the cnn_Transformer model:

```python
from cnn_Transformer import Transformer_AudioClassifier

# Instantiate the model
input_size = 16000
num_classes = 10
conv_channels = [32, 64, 128]
conv_kernel_sizes = [8, 5, 3]
conv_strides = [1, 1, 1]
embed_size = 128
num_heads = 4
num_transformer_blocks = 2
mlp_hidden_dim = 128
dropout = 0.1

model = Transformer_AudioClassifier(input_size, num_classes, conv_channels, conv_kernel_sizes, conv_strides, embed_size, num_heads, num_transformer_blocks, mlp_hidden_dim, dropout)

# Prepare input data
input_tensor = torch.randn((32, 9, 16000))

# Forward pass
output = model(input_tensor)
print(output.shape)
```

## Results

The cnn_Transformer model has been evaluated on Esc-50 urban sound dataset with various hyeprparametrs, models. Detailed results and comparisons can be found out in the jupyter notebook given.


## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch](https://pytorch.org/)
