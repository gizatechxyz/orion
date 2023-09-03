# tensor.cosh

```rust 
    fn cosh(self: @Tensor<T>) -> Tensor<T>;
```

Computes the hyperbolic cosine of all elements of the input tensor.
$$
y_i=cosh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `T` with the hyperblic cosine of the elements of the input tensor.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FixedTrait, FP8x23};

fn cosh_example() -> Tensor<FP8x23> {
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

    return tensor.cosh();
}
>>> [[8388608,12944299],[31559585,84453670]]
// The fixed point representation of
// [[, 1.54308],[3.762196, 10.067662]]
```
