# tensor.stride

```rust
fn stride(self: @Tensor<T>) -> Span<usize>;
```

Computes the stride of each dimension in the tensor.

## Args
* `self`(`@Tensor<T>`) - The input tensor.

## Returns

A span of usize representing the stride for each dimension of the tensor.

## Examples

```rust
fn stride_example() -> Span<usize> {
// We instantiate a 3D Tensor here.
// [[[0,1],[2,3]],[[4,5],[6,7]]]
let tensor = u32_tensor_2x2x2_helper();

// We can call `stride` function as follows.
return tensor.stride();
}
>>> [4,2,1]
```
