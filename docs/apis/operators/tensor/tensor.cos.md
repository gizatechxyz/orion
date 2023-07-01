#tensor.cos

```rust
fn cos(self: @Tensor<T>) -> Tensor<T>;
```

Computes the cosine of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with
the cosine value of all elements in the input tensor.

## Example

```rust
fn cos_example() -> Tensor<FixedType> {
// We instantiate a 1D Tensor here.
// tensor = [[0, 1, 2,]]
let tensor = fp8x23_tensor_1x3_helper();
let result = tensor.cos();
return result;
}
>>> [8388608,4532384,-3490893]
// The fixed point representation of
// [1, 0.5403...,-0.4161]
```
