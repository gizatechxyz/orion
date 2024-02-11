# tensor.reverse_sequence

```rust
   fn reverse_sequence(self: @Tensor<T>, sequence_lens: @Tensor<i32>, batch_axis: Option<usize>, time_axis: Option<usize>) -> 
   Tensor<T>;
```

Reverse batch of sequences having different lengths specified by sequence_lens.

* `self`(`@Array<Tensor<T>>`) - Tensor of rank r >= 2.
* `sequence_lens`(`@Tensor<T>`) - Tensor specifying lengths of the sequences in a batch. It has shape [batch_size].
* `batch_axis`(`Option<usize>`) - (Optional) Specify which axis is batch axis. Must be one of 1 (default), or 0.
* `time_axis`(`Option<usize>`) - (Optional) Specify which axis is time axis. Must be one of 0 (default), or 1.

## Panics

* Panics if the 'batch_axis' == 'time_axis'.
* Panics if the 'batch_axis' and 'time_axis' are not 0 and 1.
* Panics if the 'sequence_len' exceeding the sequence range.

## Returns

Tensor with same shape of input.

## Example
```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use core::option::OptionTrait;
fn reverse_sequence_example() -> Tensor<u32> {
    let tensor: Tensor<u32> = TensorTrait::<u32>::new(
        shape: array![4,4].span(), 
        data: array![
            0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16
            ].span(),
    );
    let sequence_lens = TensorTrait::<usize>::new(array![4].span(), array![1,2,3,4].span());
    let batch_axis = Option::Some(0);
    let time_axis = Option::Some(1);
    // We can call `split` function as follows.
    return tensor.reverse_sequence(sequence_lens, batch_axis, time_axis);
}
>>> [
        [0,1,2,3],
        [5,4,6,7],
        [10,9,8,11],
        [15,14,13,12]
    ] 
```
