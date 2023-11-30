## tensor.sequence_erase

```rust 
   fn sequence_erase(sequence: Array<Tensor<T>>, position: Option<Tensor<i32>>) -> Array<Tensor<T>>;
```

Outputs the tensor sequence with the erased tensor at the specified position.

## Args

* `tensors`(`Array<Tensor<T>>`) - The tensor sequence.
* `position`(`Option<Tensor<i32>>`) - The optional position tensor (by default erases the last tensor).

## Panics 

* Panics if position is not a scalar
* Panics if position is out of bounds [-n, n - 1]

## Returns

The tensor sequence `Array<Tensor<T>>` with the erased tensor at the specified position.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor, I32Tensor};
use orion::numbers::{i32, IntegerTrait};

fn sequence_erase_example() -> Tensor<u32> {
    let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());
    let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![4, 5, 6, 7].span());
    let tensor3 = TensorTrait::new(shape: array![2, 2].span(), data: array![8, 9, 10, 11].span());
    
    let mut sequence = ArrayTrait::new();
    sequence.append(tensor1);
    sequence.append(tensor2);
    sequence.append(tensor3);

    let position = TensorTrait::new(shape: array![].span(), data: array![IntegerTrait::new(1, false)].span());

    let result = TensorTrait::sequence_erase(sequence, position);
    return result;
}
>>> [[0, 1, 2, 3], [8, 9, 10, 11]]
```
