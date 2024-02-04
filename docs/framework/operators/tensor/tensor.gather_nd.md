# tensor.gather_nd

```rust 
   fn gather_nd(self: @Tensor<T>, indices: Tensor<T>, batch_dims: Option<usize>) -> Tensor<T>;
```

Given data tensor of rank r >= 1, indices tensor of rank q >= 1, and batch_dims integer b, this operator gathers slices of data into an output tensor of rank q + r - indices_shape[-1] - 1 - b.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `indices`(`Tensor<T>`) - Tensor of indices.
* `batch_dims`(`Option<usize>`) -  The number of batch dimensions. The gather of indexing starts from dimension of data[batch_dims:].

## Panics

* Panics if index values are not within bounds [-s, s-1] along axis of size s.
* Panics if If indices_shape[-1] > r-b.
* Panics if first b dimensions of the shape of indices tensor and data tensor are not equal.

## Returns
A new `Tensor<T>`.

## Example

```rust
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn gather_nd_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2].span(), 
        data: array![[0, 1], [2, 3]].span(), 
    );
    let indices = TensorTrait::<u32>::new(
        shape: array![4, 1].span(), 
        data: array![[0], [0], [1], [1]].span(), 
    );

    return tensor.gather_nd(
        indices: indices, 
        axis: Option::Some((0)), 
    );
}
>>> [[0, 1],
     [0, 1],
     [2, 3],
     [2, 3]]
```
