# tensor.sinh

```rust
fn sinh(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the hyperbolic sine of all elements of the input tensor.
$$
y_i=sinh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the hyperbolic sine of the elements of the input tensor.

## Examples

```rust
fn exp_example() -> Tensor<FixedType> {
// We instantiate a 2D Tensor here.
// [[0,1],[2,3]]
let tensor = u32_tensor_2x2_helper();

// We can call `sinh` function as follows.
return tensor.sinh();
}
>>> [[0,9858303],[30424311,84036026]]
// The fixed point representation of
// [[0, 1.175201],[3.62686, 10.0178749]]
```
