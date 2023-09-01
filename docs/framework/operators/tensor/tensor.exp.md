# tensor.exp

```rust 
    fn exp(self: @Tensor<T>) -> Tensor<F>;
```

Computes the exponential of all elements of the input tensor.
$$
y_i=e^{x_i}
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `F` with the exponential of the elements of the input tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_u32_fp8x23};
use orion::numbers::FP8x23;

fn exp_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),
    );

    // We can call `exp` function as follows.
    return tensor.exp();
}
>>> [[8388608,22802594],[61983844,168489688]]
// The fixed point representation of
// [[1, 2.718281],[7.38905, 20.085536]]
```
