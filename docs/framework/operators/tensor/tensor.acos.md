#tensor.acos

```rust
    fn acos(self: @Tensor<T>) -> Tensor<T>;
```

Computes the arccosine (inverse of cosine) of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the arccosine value of all elements in the input tensor.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FP8x23, FixedTrait};

fn acos_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![2].span(),
        data: array![FixedTrait::new_unscaled(0, false), FixedTrait::new_unscaled(1, false),]
            .span(),
    );

    return tensor.acos();
}
>>> [13176794, 0]
// The fixed point representation of
// [1.5707..., 0]
```
