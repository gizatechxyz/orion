# tensor.split_to_sequence

```rust 
   fn split_to_sequence(self: @Tensor<T>, axis: usize, num_outputs: Option<usize>, split: Option<Tensor<usize>>
   ) -> Array<Tensor<T>>;
```

Split a tensor into a sequence of tensors along a given axis.
## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `split`(`Option<Tensor<usize>>`) - Optional tensor (scalar or 1D) specifying the lengths of each output tensor. If not provided, the tensor is split into equal sized chunks of length 1.
* `axis`(`usize`) - The axis along which to split on.
* `keepdims` (`Option<bool>`) - If set to false, tensor shape dimension is squeezed. Default is set to true.

## Panics

* Panics if the 'axis' accepted range is not [-rank, rank-1] where r = rank(input).
* Panics if the inputed 'split' is not a scalar or 1D tensor.

## Returns

A Sequence comprising of one or more tensors (representing the splitted tensor parts).

## Examples
Case 1: split into fixed parts

```rust
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};

fn split_to_sequence_example() -> Array<Tensor<u32>> {
    let tensor: Tensor<u32> = TensorTrait::<u32>::new(
        shape: array![3,6].span(),
        data: array![0, 1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17].span(),);

    let split_t: Tensor<u32> = TensorTrait::<u32>::new(shape: array![].span(), data: array![2].span())
    // We can call `split_to_sequence` function as follows.
    return tensor.split_to_sequence(Option::Some(split_t), 1, Option::None(()));
}
>>> [
    [[0,1],[6,7],[12,13]],
    [[2,3],[8,9],[14,15]],
    [[4,5],[10,11],[16,17]]
    ]
```
Case 2: split into variable parts

```rust
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};

fn split_to_sequence_example() -> Array<Tensor<u32>> {
    let tensor: Tensor<u32> = TensorTrait::<u32>::new(
        shape: array![3,6].span(),
        data: array![0, 1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17].span(),);

    let split_t: Tensor<u32> = TensorTrait::<u32>::new(shape: array![2].span(), data: array![4,2].span())
    // We can call `split_to_sequence` function as follows.
    return tensor.split_to_sequence(Option::Some(split_t), 1, Option::None(()));
}
>>> [
     [[0,1,2,3],[6,7,8,9],[12,13,14,15]],
     [[4,5],[10,11],[16,17]]
    ]
```
Case 3: split is none whilst keepdims is set to false

```rust
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};

fn split_to_sequence_example() -> Array<Tensor<u32>> {
    let tensor: Tensor<u32> = TensorTrait::<u32>::new(
        shape: array![3,6].span(),
        data: array![0, 1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17].span(),);

   
    // We can call `split_to_sequence` function as follows.
    return tensor.split_to_sequence(Option::None(()), 0, Option::Some((false)));
}
>>> [
    [0, 1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10,11], 
    [12,13,14,15,16,17]
    ]
  
```
