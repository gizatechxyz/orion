# Neural Network

This module contains primitive Neural Net (NN) operations.

```rust
use orion::operators::nn;
```

### Data types

Orion supports currently these `NN` types.

| Data type                 | dtype                                             |
| ------------------------- | ------------------------------------------------- |
| 32-bit integer (signed)   | `Tensor<i32>`                                     |
| 8-bit integer (signed)    | `Tensor<i8>`                                      |
| 32-bit integer (unsigned) | `Tensor<u32>`                                     |
| Fixed point (signed)      | `Tensor<FP8x23 \| FP16x16 \| FP32x32 \| FP64x64>` |

### NN**Trait**

`NNTrait` contains the primitive functions to build a Neural Network.

| function | description |
| --- | --- |
| [`nn.relu`](nn.relu.md) | Applies the rectified linear unit function element-wise. |
| [`nn.leaky_relu`](nn.leaky\_relu.md) | Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise. |
| [`nn.sigmoid`](nn.sigmoid.md) | Applies the Sigmoid function to an n-dimensional input tensor. |
| [`nn.softmax`](nn.softmax.md) | Computes softmax activations. |
| [`nn.softmax_zero`](nn.softmax\_zero.md) | Computes softmax zero. |
| [`nn.logsoftmax`](nn.logsoftmax.md) | Applies the natural log to Softmax function to an n-dimensional input Tensor. |
| [`nn.softsign`](nn.softsign.md) | Applies the Softsign function element-wise. |
| [`nn.softplus`](nn.softplus.md) | Applies the Softplus function element-wise. |
| [`nn.linear`](nn.linear.md) | Performs a linear transformation of the input tensor using the provided weights and bias. |
| [`nn.hard_sigmoid`](nn.hard\_sigmoid.md) | Applies the Hard Sigmoid function to an n-dimensional input tensor. |
| [`nn.thresholded_relu`](nn.thresholded\_relu.md) | Performs the thresholded relu activation function element-wise. |
| [`nn.gemm`](nn.gemm.md) | Performs General Matrix multiplication. |
| [`nn.grid_sample`](nn.grid\_sample.md) | Computes the grid sample of the input tensor and input grid. |
| [`nn.col2im`](nn.col2im.md) | Rearranges column blocks back into a multidimensional image |
| [`nn.conv_transpose`](nn.conv\_transpose.md) | Performs the convolution transpose of the input data tensor and weight tensor. |
| [`nn.conv`](nn.conv.md) | Performs the convolution of the input data tensor and weight tensor. |
| [`nn.hard_swish`](nn.hard\_swish.md) | Applies the Hard Swish function to an n-dimensional input tensor. |

