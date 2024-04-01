# TensorTrait::tile

```rust
        fn tile(self: @Tensor<T>, repeats: Span<usize>) -> Tensor<T>;
```

Constructs a tensor by tiling a given tensor. This is the same as function tile in Numpy, but no broadcast.

 For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]

## Args

* `tensor`(`@Tensor<T>`) - Input tensor of any shape.
* `repeats`(Span<usize>) - 1D usize array of the same length as input's dimension number, includes numbers of repeated copies along input's dimensions.

## Returns

* Output tensor of the same dimensions and type as tensor input. output_dim[i] = input_dim[i] * repeats[i].

## Examples

```rust
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::I32TensorPartialEq;


fn example() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(2);
    data.append(1);
    let input_0 = TensorTrait::new(shape.span(), data.span());

    return input_0.tile(array![1, 4].span());
}
>>> [[2, 1, 2, 1, 2, 1, 2, 1]]
```
