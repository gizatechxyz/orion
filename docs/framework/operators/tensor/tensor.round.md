#tensor.round

```rust
    fn round(self: @Tensor<T>) -> Tensor<T>;
```

Computes the round value of all elements in the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the round value of all elements in the input tensor.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor};
use orion::numbers::{FixedTrait, FP16x16};

fn round_example() -> Tensor<FP16x16> {
    let tensor = TensorTrait::<FP16x16>::new(
        shape: array![3].span(),
        data: array![
            FixedTrait::new(190054, false),  // 2.9
        ]
            .span(),
    );

    return tensor.round();
}
>>> [3]
```
