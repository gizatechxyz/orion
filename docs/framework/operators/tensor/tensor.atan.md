#tensor.atan

```rust
    fn atan(self: @Tensor<T>) -> Tensor<T>;
```

Computes the arctangent (inverse of tangent) of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the arctangent (inverse of tangent) value of all elements in the input tensor.

## Example

```rust
fn atan_example() -> Tensor<FixedType> {
    // We instantiate a 1D Tensor here.
    // tensor = [0, 1, 2,]
    let tensor = fp_tensor_1x3_helper();
    let result = tensor.atan().data;
    return result;
}
>>> [0,51471,72558]
// The fixed point representation of
// [0,0.7853...,1.1071...]
```
   