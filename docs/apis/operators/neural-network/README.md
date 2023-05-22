# Neural Network

This module contains primitive Neural Net (NN) operations.

```rust
use orion::operators::nn;
```

### Data types

Orion supports currently two `NN` types.

| Data type                 | dtype    |
| ------------------------- | -------- |
| 32-bit integer (signed)   | `nn_i32` |
| 32-bit integer (unsigned) | `nn_u32` |

### NN**Trait**

```rust
use orion::operators::nn::nn_i32::NN;
// OR
use orion::operators::nn::nn_u32::NN;
```

`NNTrait` contains the primitive functions to build a Neural Network.

| function | description |
| --- | --- |
| [`nn.relu`](https://orion.gizatech.xyz/apis/operators/neural-network/nn.relu.md) | Applies the rectified linear unit function element-wise. |
| [`nn.leaky_relu`](https://orion.gizatech.xyz/apis/operators/neural-network/nn.leaky_relu.md) | Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise. |
| [`nn.softmax`](https://orion.gizatech.xyz/apis/operators/neural-network/nn.softmax.md) | Computes softmax activations. |
| [`nn.linear`](https://orion.gizatech.xyz/apis/operators/neural-network/nn.linear.md) | Performs a linear transformation of the input tensor using the provided weights and bias. |

