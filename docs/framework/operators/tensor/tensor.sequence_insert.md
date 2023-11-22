# tensor.sequence_insert

```rust 
   fn sequence_insert(self: Array<Tensor<T>>, tensor: @Tensor<T>, position: Option<Tensor<i32>>) -> Array<Tensor<T>>;
```

Returns a tensor sequence that inserts 'tensor' into 'self' at 'position'.

## Args

* `self`(`Array<Tensor<T>>`) - input sequence.
* `tensor` (`@Tensor<T>`) - the tensor to insert.
* `position` (`@Tensor<i32>`) - the index for insertion (default: -1).

## Returns

Tensor sequence containing 'tensor' inserted into 'self' at 'position'.

## Examples

Let's insert the tensor [2] into the sequence [[1], [3]] at position 1.
use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor, U32Tensor};

fn sequence_insert_example() -> Array<Tensor<u32>> {
    // Prepare sequence
    let mut sequence = ArrayTrait::new();
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(1);
    sequence.append(TensorTrait::new(shape.span(), data.span()));
    let mut data = ArrayTrait::new();
    data.append(3);

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    // Prepare input tensor
    let mut data = ArrayTrait::new();
    data.append(2);
    let tensor = TensorTrait::new(shape.span(), data.span());

    // Prepare position
    let mut shape = ArrayTrait::<usize>::new();
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 1, sign: false });
    let position = TensorTrait::<i32>::new(shape.span(), data.span())

    let sequence = self.sequence_insert(tensor, Option::Some(position));

    return sequence;
}

>>> [[1], [2], [3]]
```
