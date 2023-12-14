# tensor.acosh

```rust 
    fn acosh(self: @Tensor<T>) -> Tensor<T>;
```

Computes the inverse hyperbolic cosine of all elements of the input tensor.
$$
y_i=acosh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `T` with the hyperblic cosine of the elements of the input tensor.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FixedTrait, FP8x23};

fn acosh_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![2, 2].span(),
        data: array![
            FixedTrait::new_unscaled(1, false),
            FixedTrait::new_unscaled(2, false),
            FixedTrait::new_unscaled(3, false),
            FixedTrait::new_unscaled(4, false)
        ]
            .span(),
    );

    return tensor.acosh();
}
>>> [[0,11047444],[14786996,17309365]]
// The fixed point representation of
// [[0, 1.31696],[1.76275, 2.06344]]
```
