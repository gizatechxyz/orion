# tensor.acosh

```rust 
    fn acosh(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the inverse hyperbolic cosine of all elements of the input tensor.
$$
y_i=acosh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the hyperblic cosine of the elements of the input tensor.

## Examples

```rust
fn acosh_example() -> Tensor<FixedType> {
    // We instantiate a 2D Tensor here.
    // [[1,2],[3,4]]
    let tensor = u32_tensor_2x2_helper();
		
    // We can call `acosh` function as follows.
    return tensor.acosh();
}
>>> [[0,11047444],[14786996,17309365]]
// The fixed point representation of
// [[0, 1.31696],[1.76275, 2.06344]]
```
