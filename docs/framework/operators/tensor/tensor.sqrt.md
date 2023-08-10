#tensor.sqrt

```rust
    fn sqrt(self: @Tensor<T>) -> Tensor<T>;
```

Computes the square root of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the arctangent (inverse of tangent) value of all elements in the input tensor.
fn sqrt_example() -> Tensor<FixedType> {
    // We instantiate a 1D Tensor here.
    // tensor = [0, 1, 2]
    let tensor = fp_tensor_1x3_helper();
    let result = tensor.sqrt().data;
    return result;
}
>>> [0,65536,92672]
// The fixed point representation of
// [0,1,1.4142...]
```
   