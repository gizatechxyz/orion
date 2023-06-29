# Compatibility

To see the full list of available ONNX Operators refer to [this table](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

You can see below the list of current supported ONNX Operators:

|                          Operator                           |     Implemented      |
| :---------------------------------------------------------: | :------------------: |
|         [MatMul](operators/tensor/tensor.matmul.md)         | :white\_check\_mark: |
|     [MatMulInteger](operators/tensor/tensor.matmul.md)      | :white\_check\_mark: |
|       [Add](operators/tensor/#arithmetic-operations)        | :white\_check\_mark: |
|       [Sub](operators/tensor/#arithmetic-operations)        | :white\_check\_mark: |
|       [Mul](operators/tensor/#arithmetic-operations)        | :white\_check\_mark: |
|       [Div](operators/tensor/#arithmetic-operations)        | :white\_check\_mark: |
|           [Equal](operators/tensor/tensor.eq.md)            | :white\_check\_mark: |
|        [Greater](operators/tensor/tensor.greater.md)        | :white\_check\_mark: |
| [GreaterOrEqual](operators/tensor/tensor.greater\_equal.md) | :white\_check\_mark: |
|           [Less](operators/tensor/tensor.less.md)           | :white\_check\_mark: |
|    [LessOrEqual](operators/tensor/tensor.less\_equal.md)    | :white\_check\_mark: |
|            [Abs](operators/tensor/tensor.abs.md)            | :white\_check\_mark: |
|           [Ceil](operators/tensor/tensor.ceil.md)           | :white\_check\_mark: |
|            [Exp](operators/tensor/tensor.exp.md)            | :white\_check\_mark: |
|             [Ln](operators/tensor/tensor.ln.md)             | :white\_check\_mark: |
|        [Reshape](operators/tensor/tensor.reshape.md)        | :white\_check\_mark: |
|      [Transpose](operators/tensor/tensor.transpose.md)      | :white\_check\_mark: |
|         [ArgMax](operators/tensor/tensor.argmax.md)         | :white\_check\_mark: |
|         [ArgMin](operators/tensor/tensor.argmin.md)         | :white\_check\_mark: |
|     [ReduceSum](operators/tensor/tensor.reduce\_sum.md)     | :white\_check\_mark: |
|         [CumSum](operators/tensor/tensor.cumsum.md)         | :white\_check\_mark: |
|         [Relu](operators/neural-network/nn.relu.md)         | :white\_check\_mark: |
|   [LeakyRelu](operators/neural-network/nn.leaky\_relu.md)   | :white\_check\_mark: |
|      [Sigmoid](operators/neural-network/nn.sigmoid.md)      | :white\_check\_mark: |
|      [Softmax](operators/neural-network/nn.softmax.md)      | :white\_check\_mark: |
|   [LogSoftmax](operators/neural-network/nn.logsoftmax.md)   | :white\_check\_mark: |
|     [Softsign](operators/neural-network/nn.softsign.md)     | :white\_check\_mark: |
|     [Softplus](operators/neural-network/nn.softplus.md)     | :white\_check\_mark: |
|       [Linear](operators/neural-network/nn.linear.md)       | :white\_check\_mark: |
|           [Sinh](operators/tensor/tensor.sinh.md)           | :white\_check\_mark: |
|           [Cosh](operators/tensor/tensor.cosh.md)           | :white\_check\_mark: |


Performance optimizations:

|    Optimization    |     Implemented      |
| :----------------: | :------------------: |
| 8-bit quantization | :white\_check\_mark: |

Current Operators support: **28/156 (18%)**