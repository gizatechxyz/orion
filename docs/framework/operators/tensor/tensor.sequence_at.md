## tensor.sequence_at

```rust 
   fn sequence_at(sequence: Array<Tensor<T>>, position: Tensor<i32>) -> Tensor<T>;
```

Outputs the tensor at the specified position in the input sequence.

## Args

* `tensors`(`Array<Tensor<T>>`) - The tensor sequence.
* `position`(`Tensor<i32>`) - The position tensor.

## Panics 

* Panics if position is not a scalar
* Panics if position is out of bounds [-n, n - 1]

## Returns

The tensor `Tensor<T>` from the sequence at the specified position.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor, I32Tensor};
use orion::numbers::{i32, IntegerTrait};

fn sequence_at_example() -> Tensor<u32> {
    let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());
    let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![4, 5, 6, 7].span());
    
    let mut sequence = ArrayTrait::new();
    sequence.append(tensor1);
    sequence.append(tensor2);

    let position = TensorTrait::new(shape: array![].span(), data: array![IntegerTrait::new(1, false)].span());

    let result = TensorTrait::sequence_at(sequence, position);
    return result;
}
>>> [4, 5, 6, 7]
```
