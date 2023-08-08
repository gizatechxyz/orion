#tensor.acos

```rust
fn acos(self: @Tensor<T>) -> Tensor<T>;
```

Computes the arccosine (inverse of cosine) of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with
the arccosine value of all elements in the input tensor.

## Example

```rust
fn acos_example() -> Tensor<FixedType> {
// We instantiate a 1D Tensor here.
// tensor = [[0, 1]]
let tensor = fp8x23_tensor_1x2_helper();
let result = tensor.acos();
return result;
}
>>> [13176794, 0]
// The fixed point representation of
// [1.5707..., 0]
```
