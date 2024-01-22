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
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
use orion::operators::nn::{NNTrait, I32NN};

fn relu_example() -> Tensor<i32> {
    let tensor = TensorTrait::<i32>::new(
        shape: array![2, 2].span(),
        data: array![1, 2, -1, -2].span(),
    );

    return NNTrait::relu(@tensor);
}
>>> [[1,2],[0,0]]
```
