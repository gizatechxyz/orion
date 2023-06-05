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
fn exp_example() -> Tensor<FixedType> {
// We instantiate a 2D Tensor here.
// [[0,1],[2,3]]
let tensor = u32_tensor_2x2_helper();

// We can call `exp` function as follows.
return tensor.exp();
}
>>> [[8388608,22802594],[61983844,168489688]]
// The fixed point representation of
// [[1, 2.718281],[7.38905, 20.085536]]
```
