# tensor.log

```rust 
    fn log(self: @Tensor<T>) -> Tensor<F>;
```

Computes the natural log of all elements of the input tensor.
$$
y_i=log({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `F` with the natural log of the elements of the input tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_u32_fp8x23};
use orion::numbers::FP8x23;

fn log_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2].span(), data: array![1, 2, 3, 100].span(),
    );

    // We can call `log` function as follows.
    return tensor.log();
}
>>> [[0, 5814538, 9215825, 38630966]]
// The fixed point representation of
/// [[0, 0.693147, 1.098612, 4.605170]]
```
