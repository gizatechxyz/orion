# tensor.split

```rust 
   fn split(self: @Tensor<T>, axis: usize, num_outputs: Option<usize>, split: Option<Tensor<usize>>
   ) -> Array<Tensor<T>>;
```
## Args
Split a tensor into a list of tensors, along the specified ‘axis’

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The axis along which to split on.
* `num_outputs `(Option<usize>) - Number of outputs to split parts of the tensor into. 
* `split  `(Option<Tensor<usize>>) - Optional length of each output.

## Panics

* Panics if the 'axis' accepted range is not [-rank, rank-1] where r = rank(input).
* Panics if the 'split' values not >= 0. Sum of the values is not equal to the dim value at ‘axis’ specified.
* Panics if the input 'split' or the attribute 'num_outputs' both are specified or not.

## Returns

One or more outputs forming list of tensors after splitting.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use core::option::OptionTrait;
fn split_tensor_example() -> Array<Tensor<u32>> {
    let tensor: Tensor<u32> = TensorTrait::<u32>::new(
        shape: array![2,4].span(), 
        data: array![
            0, 1, 2, 3, 4, 5, 6, 7
            ].span(),
    );
    let num_outputs = Option::Some(2);
    // split = Option::Some(array![1, 1].span());
    let split_num: Option<Tensor<usize>> = Option::None(());
    // We can call `split` function as follows.
    return tensor.split(1, num_outputs, split_num);
}
>>> [[0,1],[4,5]]
    [[2,3],[6,7]]
```
