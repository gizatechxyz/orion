# tensor.sign

```rust 
   fn sign(self: @Tensor<T>) -> Tensor<T>;
```

Calculates the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.

## Args

* `self`(`@Tensor<T>`) - Tensor of data to calculates the sign of the given input tensor element-wise.

## Returns 

A new `Tensor<T>` of the same shape as the input tensor with The sign of the input tensor computed element-wise.

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};

fn sign_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![11].span(), 
        data: array![-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].span(), 
    );

    return tensor.sign();
}
>>> [-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1]
```
