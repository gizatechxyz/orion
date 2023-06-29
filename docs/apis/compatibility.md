# Compatibility

To see the full list of available ONNX Operators refer to [this table](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

You can see below the list of current supported ONNX Operators:

|                          Operator                          |    Implemented     |
| :--------------------------------------------------------: | :----------------: |
|        [MatMul](operators/tensor/tensor.matmul.md)         | :white_check_mark: |
|     [MatMulInteger](operators/tensor/tensor.matmul.md)     | :white_check_mark: |
|       [Add](operators/tensor/#arithmetic-operations)       | :white_check_mark: |
|       [Sub](operators/tensor/#arithmetic-operations)       | :white_check_mark: |
|       [Mul](operators/tensor/#arithmetic-operations)       | :white_check_mark: |
|       [Div](operators/tensor/#arithmetic-operations)       | :white_check_mark: |
|           [Equal](operators/tensor/tensor.eq.md)           | :white_check_mark: |
|       [Greater](operators/tensor/tensor.greater.md)        | :white_check_mark: |
| [GreaterOrEqual](operators/tensor/tensor.greater_equal.md) | :white_check_mark: |
|          [Less](operators/tensor/tensor.less.md)           | :white_check_mark: |
|    [LessOrEqual](operators/tensor/tensor.less_equal.md)    | :white_check_mark: |
|           [Abs](operators/tensor/tensor.abs.md)            | :white_check_mark: |
|          [Ceil](operators/tensor/tensor.ceil.md)           | :white_check_mark: |
|           [Exp](operators/tensor/tensor.exp.md)            | :white_check_mark: |
|            [Ln](operators/tensor/tensor.ln.md)             | :white_check_mark: |
|       [Reshape](operators/tensor/tensor.reshape.md)        | :white_check_mark: |
|     [Transpose](operators/tensor/tensor.transpose.md)      | :white_check_mark: |
|        [ArgMax](operators/tensor/tensor.argmax.md)         | :white_check_mark: |
|        [ArgMin](operators/tensor/tensor.argmin.md)         | :white_check_mark: |
|     [ReduceSum](operators/tensor/tensor.reduce_sum.md)     | :white_check_mark: |
|        [CumSum](operators/tensor/tensor.cumsum.md)         | :white_check_mark: |
|        [Relu](operators/neural-network/nn.relu.md)         | :white_check_mark: |
|   [LeakyRelu](operators/neural-network/nn.leaky_relu.md)   | :white_check_mark: |
|     [Sigmoid](operators/neural-network/nn.sigmoid.md)      | :white_check_mark: |
|     [Softmax](operators/neural-network/nn.softmax.md)      | :white_check_mark: |
|  [LogSoftmax](operators/neural-network/nn.logsoftmax.md)   | :white_check_mark: |
|    [Softsign](operators/neural-network/nn.softsign.md)     | :white_check_mark: |
|    [Softplus](operators/neural-network/nn.softplus.md)     | :white_check_mark: |
|      [Linear](operators/neural-network/nn.linear.md)       | :white_check_mark: |
|          [Sinh](operators/tensor/tensor.sinh.md)           | :white_check_mark: |
|          [Tanh](operators/tensor/tensor.tanh.md)           | :white_check_mark: |

Performance optimizations:

|    Optimization    |    Implemented     |
| :----------------: | :----------------: |
| 8-bit quantization | :white_check_mark: |

Current Operators support: **29/156 (19%)**
