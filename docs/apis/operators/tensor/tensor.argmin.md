# tensor.argmin

```rust
fn argmin(self: @Tensor<T>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>) -> Tensor<usize>;
```

Returns the index of the minimum value along the specified axis.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The axis along which to compute the argmin.
* `keepdims`(`Option<bool>`) - If true, retains reduced dimensions with length 1. Defaults to true.
* `select_last_index`(`Option<bool>`) - If true, the index of the last occurrence of the minimum value is returned. Defaults to false.

## Panics

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns

A new `Tensor<T>` instance containing the indices of the minimum values along the specified axis.

## Examples

Case 1: argmin with default parameters

```rust
fn argmin_example() -> Tensor<usize> {
// We instantiate a 3D Tensor here.
// [[[0,1],[2,3]],[[4,4],[5,5]]]
let tensor = u32_tensor_2x2x2_helper();

// We can call `argmin` function as follows.
return tensor.argmin(2,Option::None(()),Option::None(()));
}
>>> [[[0,0],[0,0]]]

```
Case 2: argmin with keepdims set to false

```rust
fn argmin_example() -> Tensor<usize> {
// We instantiate a 3D Tensor here.
// [[[0,1],[2,3]],[[4,4],[5,5]]]
let tensor = u32_tensor_2x2x2_helper();

// We can call `argmin` function as follows.
return tensor.argmin(2,Option::Some(false),Option::None(()));
}
>>> [[0,0],[0,0]]
```

Case 3: argmin with select_last_index set to true

```rust
fn argmin_example() -> Tensor<usize> {
// We instantiate a 3D Tensor here.
// [[[0,1],[2,3]],[[4,4],[5,5]]]
let tensor = u32_tensor_2x2x2_helper();

// We can call `argmin` function as follows.
return tensor.argmin(2,Option::None(()),Option::Some(true));
}
>>> [[[0,0],[1,1]]]
```
