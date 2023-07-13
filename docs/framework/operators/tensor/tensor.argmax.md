# tensor.argmax

```rust 
   fn argmax(self: @Tensor<T>, axis: usize, keepdims: Option<bool>, select_last_index: Option<bool>) -> Tensor<usize>;
```

Returns the index of the maximum value along the specified axis.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The axis along which to compute the argmax.
* `keepdims`(`Option<bool>`) - If true, retains reduced dimensions with length 1. Defaults to true.
* `select_last_index`(`Option<bool>`) - If true, the index of the last occurrence of the maximum value is returned. Defaults to false.   

## Panics

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns 

A new `Tensor<T>` instance containing the indices of the maximum values along the specified axis.

## Examples

Case 1: argmax with default parameters

```rust
fn argmax_example() -> Tensor<usize> {
    // We instantiate a 3D Tensor here.
    // [[[0,1],[2,3]],[[4,4],[5,5]]]
    let tensor = u32_tensor_2x2x2_helper();
		
    // We can call `argmax` function as follows.
    return tensor.argmax(2,Option::None(()),Option::None(()));
}
>>> [[[1,1],[0,0]]]

```
Case 2: argmax with keepdims set to false

```rust
fn argmax_example() -> Tensor<usize> {
    // We instantiate a 3D Tensor here.
    // [[[0,1],[2,3]],[[4,4],[5,5]]]
    let tensor = u32_tensor_2x2x2_helper();

    // We can call `argmax` function as follows.
    return tensor.argmax(2,Option::Some(false),Option::None(()));
}
>>> [[1,1],[0,0]]
```

Case 3: argmax with select_last_index set to true

```rust
fn argmax_example() -> Tensor<usize> {
    // We instantiate a 3D Tensor here.
    // [[[0,1],[2,3]],[[4,4],[5,5]]]
    let tensor = u32_tensor_2x2x2_helper();

    // We can call `argmax` function as follows.
    return tensor.argmax(2,Option::None(()),Option::Some(true));
}
>>> [[[1,1],[1,1]]]
```
