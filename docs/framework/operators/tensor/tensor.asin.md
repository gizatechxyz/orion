#tensor.asin

```rust
    fn asin(self: @Tensor<T>) -> Tensor<T>;
```

Computes the arcsine (inverse of sine) of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the arcsine value of all elements in the input tensor.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_fp8x23};
use orion::numbers::{FixedTrait, FP8x23};

fn asin_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![2].span(),
        data: array![FixedTrait::new_unscaled(0, false), FixedTrait::new_unscaled(1, false),]
            .span(),
    );

    return tensor.asin();
}
>>> [0, 13176794]
// The fixed point representation of
// [0, 1.5707...]
```
