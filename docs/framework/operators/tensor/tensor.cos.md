#tensor.cos

```rust
    fn cos(self: @Tensor<T>) -> Tensor<T>;
```

Computes the cosine of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the cosine value of all elements in the input tensor.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_fp8x23};
use orion::numbers::{FP8x23, FixedTrait};

fn cos_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![3].span(),
        data: array![
            FixedTrait::new_unscaled(0, false),
            FixedTrait::new_unscaled(1, false),
            FixedTrait::new_unscaled(2, false)
        ]
            .span(),
    );

    return tensor.cos();
}
>>> [8388608,4532384,-3490893]
// The fixed point representation of
// [1, 0.5403...,-0.4161]
```
