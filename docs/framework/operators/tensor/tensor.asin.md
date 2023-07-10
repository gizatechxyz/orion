#tensor.asin

```rust
fn asin(self: @Tensor<T>) -> Tensor<T>;
```

Computes the arcsine (inverse of sine) of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with
the arcsine value of all elements in the input tensor.

## Example

```rust
fn asin_example() -> Tensor<FixedType> {
// We instantiate a 1D Tensor here.
// tensor = [[0, 1]]
let tensor = fp8x23_tensor_1x2_helper();
let result = tensor.asin();
return result;
}
>>> [0, 13176794]
// The fixed point representation of
// [0, 1.5707...]
```
