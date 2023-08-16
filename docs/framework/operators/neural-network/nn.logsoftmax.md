# NNTrait::logsoftmax

```rust 
   fn logsoftmax(tensor: @Tensor<T>, axis: usize) -> Tensor<FixedType>
```

Applies the natural log to Softmax function to an n-dimensional input Tensor consisting of values in the range \[0,1].

$$
\text{log softmax}(x_i) = \log(frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}})
$$

## Args

* `tensor`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The axis along which to compute the natural lof softmax outputs.

## Returns

A Tensor of fixed point numbers with the same shape than the input Tensor.

## Examples

```rust
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

fn logsoftmax_example() -> Tensor<FixedType> {
    // We instantiate a 2D Tensor here.
    // [[0,1],[2,3]]
    let tensor = u32_tensor_2x2_helper();
		
    // We can call `logsoftmax` function as follows.
    return NNTrait::logsoftmax(@tensor, 1);
}
    This will first generate the softmax output tensor
>>> [[2255697,6132911],[2255697,6132911]]
    // The fixed point representation of
    // [[0.2689, 0.7311],[0.2689, 0.7311]]
    
    Applying the natural log to this tensor yields
>>> 
    // The fixed point representation of:
    // [[-1.3134, -0.3132],[-1.3134, -0.3132]]
```
