# tensor.tanh

```rust 
    fn tanh(self: @Tensor<T>) -> Tensor<F>;
```

Computes the hyperbolic tangent of all elements of the input tensor.
$$
y_i=tanh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `F` with the hyperbolic tangent of the elements of the input tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_fp8x23};
use orion::numbers::{FixedTrait, FP8x23};

fn tanh_example() -> Tensor<FP8x23> {
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

    return tensor.tanh();
}
>>> [[0,6388715],[8086850,8347125]]
// The fixed point representation of
// [[0, 0.761594],[0.96403, 0.9951]]
```
