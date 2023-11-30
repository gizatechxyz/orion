# tensor.concat_from_sequence

```rust 
   fn concat_from_sequence(sequence: Array<Tensor<T>>, axis: i32, new_axis: Option<usize>) -> Tensor<T>;
```

Concatenate a sequence of tensors into a single tensor.

## Args

* `sequence`(`Array<Tensor<T>>`) - The input sequence.
* `axis`(`i32`) -  Axis to concat on.
* `new_axis`(`Option<usize>`) -  Optionally added new axis.

## Panics

* Panics if new_axis not 0 or 1 (if value provided).
* Panics if axis not in accepted ranges.
* Panics if sequence length is not greater than 1.

## Returns 

A new `Tensor<T>` concatenated tensor from the input tensor sequence.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn concat_example() -> Tensor<u32> {
    let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);

    let mut sequence = ArrayTrait::new();
    sequence.append(tensor1);
    sequence.append(tensor2);

    let result = TensorTrait::concat_from_sequence(sequence: sequence, axis: 0, new_axis: Option::Some(0));
    return result;
}
>>> [[0. 1.]
     [2. 3.],
     [0. 1.]
     [2. 3.]]

    result.shape
>>> (4, 2)

   let result = TensorTrait::concat_from_sequence(sequence: sequence, axis: 1, new_axis: Option::Some(0));
   return result;
}
>>> [[0. 1., 0., 1.]
     [2. 3., 2., 3.]]

    result.shape
>>> (2, 4 ) 
```
