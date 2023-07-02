# tensor.tanh

```rust
fn tanh(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the exponential of all elements of the input tensor.
$$
y_i=tanh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the exponential of the elements of the input tensor.

## Examples

```rust
fn tanh_example() -> Tensor<FixedType> {
// We instantiate a 2D Tensor here.
// [[0,1],[2,3]]
let tensor = u32_tensor_2x2_helper();

// We can call `tanh` function as follows.
return tensor.tanh();
}
>>> [[0,6388715],[8086850,8347125]]
// The fixed point representation of
// [[0, 0.761594],[0.96403, 0.9951]]
```
