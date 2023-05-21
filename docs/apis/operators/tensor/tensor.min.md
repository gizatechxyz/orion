# tensor.min

```rust
fn min(self: @Tensor<T>) -> T;
```

Returns the minimum value in the tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

The minimum `T` value in the tensor.

## Examples

```rust
fn min_example() -> u32 {
// We instantiate a 3D Tensor here.
// [[[0,1],[2,3]],[[4,5],[6,7]]]
let tensor = u32_tensor_2x2x2_helper();

// We can call `min` function as follows.
return tensor.min();
}
>>> 0
```
