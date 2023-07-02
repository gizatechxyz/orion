# tensor.asinh

```rust
fn asinh(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the inverse hyperbolic sine of all elements of the input tensor.
$$
y_i=asinh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the hyperblic cosine of the elements of the input tensor.

## Examples

```rust
fn asinh_example() -> Tensor<FixedType> {
// We instantiate a 2D Tensor here.
// [[0,1],[2,3]]
let tensor = u32_tensor_2x2_helper();

// We can call `asinh` function as follows.
return tensor.asinh();
}
>>> [[0,7393498],[12110093,15254235]]
// The fixed point representation of
// [[0, 0.8814],[1.44364, 1.8185]]
```
