# NNTrait::relu

```rust 
   fn relu(tensor: @Tensor<T>) -> Tensor<T>;
```

Applies the rectified linear unit function element-wise

$$
ReLU(x)=(x)^+=max(0,x)
$$

## Args

* `tensor`(`@Tensor<T>`) - The input tensor.

## Returns

A `Tensor<T>` with the same shape as the input tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
use orion::numbers::signed_integer::i32::{i32, IntegerTrait};

fn relu_example() -> Tensor<i32> {
    let tensor = TensorTrait::<i32>::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::new(1, false),
            IntegerTrait::new(2, false),
            IntegerTrait::new(1, true),
            IntegerTrait::new(2, true),
        ].span(),
        extra: Option::None(())
    );

    return NNTrait::relu(@tensor);
}
>>> [[1,2],[0,0]]
```
