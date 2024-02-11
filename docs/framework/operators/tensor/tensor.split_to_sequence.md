# tensor.split_to_sequence

```rust 
   fn split_to_sequence(
       self: @Tensor<T>, axis: usize, keepdims: usize, split: Option<Tensor<usize>>
   ) -> Array<Tensor<T>>;
```

Split a tensor into a sequence of tensors, along the specified ‘axis’


## Args
* `self`(`@Tensor<T>`) - The input tensor to split.
* `axis`(`usize`) - The axis along which to split on.
* `keepdims  `(`usize`) - Keep the split dimension or not. If input ‘split’ is specified, this attribute is ignored.
* `split  `(`Option<Tensor<usize>>`) - Length of each output. It can be either a scalar(tensor of empty shape), or a 1-D tensor. All values must be >= 0.

## Panics

* Panics if the 'axis' accepted range is not [-rank, rank-1] where r = rank(input).
* Panics if the 'split' is not either a scalar (tensor of empty shape), or a 1-D tensor.

## Returns

One or more outputs forming a sequence of tensors after splitting.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use core::option::OptionTrait;
fn split_to_sequence_example() -> Array<Tensor<u32>> {
    let tensor: Tensor<u32> = TensorTrait::<u32>::new(
        shape: array![2,4].span(), 
        data: array![
            0, 1, 2, 3, 4, 5, 6, 7
            ].span(),
    );
    let num_outputs = Option::Some(2);
    // let split = Option::Some(TensorTrait::new(array![1].span(), array![2].span()));
    let split: Option<Tensor<usize>> = Option::Some(TensorTrait::new(array![2].span(), array![2, 2].span()));
    // We can call `split_to_sequence` function as follows.
    return tensor.split_to_sequence(1, 1, split);
}
>>> [
        [[0,1],[4,5]],
        [[2,3],[6,7]]
    ]
```
