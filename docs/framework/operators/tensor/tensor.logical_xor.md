#tensor.logical_xor

```rust
    fn logical_xor(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
```

Computes the logical XOR of two tensors element-wise.
The input tensors must have either:
* Exactly the same shape
* The same number of dimensions and the length of each dimension is either a common length or 1.

## Args

* `self`(`@Tensor<T>`) - The first tensor to be compared
* `other`(`@Tensor<T>`) - The second tensor to be compared

## Panics

* Panics if the shapes are not equal or broadcastable

## Returns

A new `Tensor<usize>` of booleans (0 or 1) with the same shape as the broadcasted inputs.

## Examples

Case 1: Compare tensors with same shape

```rust
fn logical_xor_example() -> Tensor<usize> {
    // We instantiate two 3D Tensors here
    // tensor_y = [[0,1,2],[3,4,5],[6,7,8]]
    // tensor_z = [[0,1,2],[3,4,5],[9,1,5]]
    let tensor_y = u32_tensor_2x2x2_helper();
    let tensor_z = u32_tensor_2x2x2_helper();
    let result = tensor_y.logical_xor(@tensor_z);
    return result;
}
>>> [0,0,0,0,0,0,0,0,0]
```

Case 2: Compare tensors with different shapes

```rust
fn logical_xor_example() -> Tensor<usize> {
    // tensor_y = [[0,1,2],[3,4,5],[6,7,8]]
    // tensor_z = [[0,1,2]]
    let tensor_y = u32_tensor_3x3_helper();
    let tensor_z = u32_tensor_3x1_helper();
    let result = tensor_y.logical_xor(@tensor_z);
    // We could equally do something like:
    // let result = tensor_z.logical_xor(@tensor_y);
    return result;
}
>>> [0,0,0,1,0,0,1,0,0]
```
