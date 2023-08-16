# NNTrait::leaky_relu

```rust
 fn leaky_relu(inputs: @Tensor<T>, alpha: @FixedType) -> Tensor<FixedType>
```

Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise to a given tensor.

The Leaky ReLU function is defined as f(x) = alpha * x if x < 0, f(x) = x otherwise, where x is the input element.

## Args
* `inputs`(`@Tensor<T>`) - A snapshot of a tensor to which the Leaky ReLU function will be applied.
* `alpha`(`@FixedType`) - A snapshot of a FixedType scalar that defines the alpha value of the Leaky ReLU function.

## Panics

* Panics if gas limit is exceeded during execution.

## Returns
A new FixedType tensor with the same shape as the input tensor and the Leaky ReLU function applied element-wise.

## Examples

```rust
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;

fn leaky_relu_example() -> Tensor<u32> {
    // We instantiate a 2D Tensor here.
    // [[1,2,-1],[-2,0,0]]
    let tensor = i32_tensor_2x3_helper();
    let alpha = Fixed::new(6710886_u128, false); // 0.1
		
    // We can call `leaky_relu` function as follows.
    return NNTrait::leaky_relu(@tensor, @alpha);
}
```
