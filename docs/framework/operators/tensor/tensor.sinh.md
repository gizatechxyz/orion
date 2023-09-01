# tensor.sinh

```rust 
    fn sinh(self: @Tensor<T>) -> Tensor<F>;
```

Computes the hyperbolic sine of all elements of the input tensor.
$$
y_i=sinh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `F` with the hyperbolic sine of the elements of the input tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_fp8x23};
use orion::numbers::{FixedTrait, FP8x23};

fn sinh_example() -> Tensor<FP8x23> {
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

    return tensor.sinh();
}
>>> [[0,9858303],[30424311,84036026]]
// The fixed point representation of
// [[0, 1.175201],[3.62686, 10.0178749]]
```
