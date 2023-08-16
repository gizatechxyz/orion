# tensor.log

```rust 
    fn log(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the natural log of all elements of the input tensor.
$$
y_i=log({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the natural log of the elements of the input tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};

fn log_example() -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2].span(),
        data: array![1, 2, 3, 100].span(),
        extra: Option::Some(extra)
    );

    // We can call `log` function as follows.
    return tensor.log();
}
>>> [[0, 5814538, 9215825, 38630966]]
// The fixed point representation of
/// [[0, 0.693147, 1.098612, 4.605170]]
```
