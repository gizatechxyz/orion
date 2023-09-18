# tensor.asinh

```rust 
    fn asinh(self: @Tensor<T>) -> Tensor<T>;
```

Computes the inverse hyperbolic sine of all elements of the input tensor.
$$
y_i=asinh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `T` with the hyperblic sine of the elements of the input tensor.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FixedTrait, FP8x23};

fn asinh_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![2, 2].span(),
        data: array![
            FixedTrait::new_unscaled(0, false),
            FixedTrait::new_unscaled(1, false),
            FixedTrait::new_unscaled(2, false),
            FixedTrait::new_unscaled(3, false)
        ]
            .span(),
    );

    return tensor.asinh();
}
>>> [[0,7393498],[12110093,15254235]]
// The fixed point representation of
// [[0, 0.8814],[1.44364, 1.8185]]
```
