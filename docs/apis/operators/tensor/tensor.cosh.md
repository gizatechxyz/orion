# tensor.cosh

```rust
fn cosh(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the hyperbolic cosine of all elements of the input tensor.
$$
y_i=cosh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the hyperblic cosine of the elements of the input tensor.

## Examples

```rust
fn exp_example() -> Tensor<FixedType> {
// We instantiate a 2D Tensor here.
// [[0,1],[2,3]]
let tensor = u32_tensor_2x2_helper();

// We can call `cosh` function as follows.
return tensor.cosh();
}
>>> [[9858303,12944299],[31559585,84453670]]
// The fixed point representation of
// [[0, 1.54308],[3.762196, 10.067662]]
```
