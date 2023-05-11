# Neural Network

This module contains primitive Neural Net (NN) operations.

```rust
use onnx_cairo::operators::nn;
```

### Data types

ONNX-Cairo supports currently two `NN` types.

| Data type                 | dtype    |
| ------------------------- | -------- |
| 32-bit integer (signed)   | `nn_i32` |
| 32-bit integer (unsigned) | `nn_u32` |

### NN Module

```rust
use onnx_cairo::operators::nn::nn_i32::NN;
// OR 
use onnx_cairo::operators::nn::nn_u32::NN;
```

`NN` module contains the primitive functions to build a Neural Network.

| function                   | description                                             |
| -------------------------- | ------------------------------------------------------- |
| [`relu`](nn-relu.md)       | Applies the rectified linear unit function element-wise |
| [`softmax`](nn-softmax.md) | Computes softmax activations.                           |
