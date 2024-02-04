# tensor.log

```rust 
    fn log(self: @Tensor<T>) -> Tensor<T>;
```

Computes the natural log of all elements of the input tensor.
$$
y_i=log({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `T` with the natural log of the elements of the input tensor.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FP8x23, FixedTrait};

fn log_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![2, 2].span(), 
        data: array![
                FixedTrait::new_unscaled(0, false), 
                FixedTrait::new_unscaled(1, false), 
                FixedTrait::new_unscaled(2, false), 
                FixedTrait::new_unscaled(100, false), 
            ]
    );

    // We can call `log` function as follows.
    return tensor.log();
}
>>> [[0, 5814538, 9215825, 38630966]]
// The fixed point representation of
/// [[0, 0.693147, 1.098612, 4.605170]]
```
