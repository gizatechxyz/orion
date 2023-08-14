# tensor.exp

```rust 
    fn exp(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the exponential of all elements of the input tensor.
$$
y_i=e^{x_i}
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the exponential of the elements of the input tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::numbers::fixed_point::core::{FixedType, FixedImpl};
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32};

fn exp_example() -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2].span(),
        data: array![0, 1, 2, 3].span(),
        extra: Option::Some(extra)
    );

    // We can call `exp` function as follows.
    return tensor.exp();
}
>>> [[8388608,22802594],[61983844,168489688]]
// The fixed point representation of
// [[1, 2.718281],[7.38905, 20.085536]]
```
