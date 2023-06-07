#tensor.ceil

```rust
fn ceil(self: @Tensor<T>) -> Tensor<T>;
```

Rounds up the value of each element in the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with
the rounded up value of all elements in the input tensor.

## Example

```rust
fn ceil_example() -> Tensor<FixedType> {
// We instantiate a 3D Tensor here.
// tensor = [[0,0.003576,11.9999947548,-11.9999947548]]
let tensor = fp8x23_tensor_1x4_helper();
let result = tensor.ceil();
return result;
}
>>> [0,1,12,-11]
```
