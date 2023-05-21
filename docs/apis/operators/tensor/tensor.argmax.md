# tensor.argmax

```rust
fn argmax(self: @Tensor<T>, axis: usize) -> Tensor<usize>;
```

Returns the index of the maximum value along the specified axis.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The axis along which to compute the argmax.

## Panics

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns

A new `Tensor<T>` instance containing the indices of the maximum values along the specified axis.

## Examples

```rust
fn argmax_example() -> Tensor<usize> {
// We instantiate a 3D Tensor here.
// [[[0,1],[2,3]],[[4,5],[6,7]]]
let tensor = u32_tensor_2x2x2_helper();

// We can call `argmax` function as follows.
return tensor.argmax(0);
}
>>> [[1,1],[1,1]]
```
