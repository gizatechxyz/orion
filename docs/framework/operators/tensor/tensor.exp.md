# tensor.exp

```rust 
    fn exp(self: @Tensor<T>) -> Tensor<T>;
```

Computes the exponential of all elements of the input tensor.
$$
y_i=e^{x_i}
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `T` with the exponential of the elements of the input tensor.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FP8x23, FixedTrait};

fn exp_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2].span(), 
        data: array![
                FixedTrait::new_unscaled(0, false), 
                FixedTrait::new_unscaled(1, false), 
                FixedTrait::new_unscaled(2, false), 
                FixedTrait::new_unscaled(3, false), 
            ]
    );

    // We can call `exp` function as follows.
    return tensor.exp();
}
>>> [[8388608,22802594],[61983844,168489688]]
// The fixed point representation of
// [[1, 2.718281],[7.38905, 20.085536]]
```
