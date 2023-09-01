# tensor.flatten

```rust 
   fn flatten(self: @Tensor<T>, axis: usize) -> Tensor<T>;
```

Flattens the input tensor into a 2D tensor. 
If input tensor has shape (1, 2, 3,...n) then the output will have shape
(1 * 2 * 3 * ... (axis-1), axis * (axis+1) * ... n).

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - Indicate up to which input dimensions (exclusive) should be flattened. 

## Panics

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns 

A new `Tensor<T>` instance containing the flattened version of the input tensor.

## Examples

Case 1: flatten with axis 0

```rust
fn flatten_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(),
        data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
        extra: Option::None(())
    );

    return tensor.flatten(0); // equivalent to tensor.reshape(1,8)
}
>>> [[0,1,2,5,4,9,6,13]]
```

Case 2: flatten with axis 1

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_u32_fp8x23};

fn flatten_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    return tensor.flatten(1); // equivalent to tensor.reshape(2,4)
}
>>> [[0,1,2,3],[4,5,6,7]]
```

Case 3: flatten with axis 2

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_u32_fp8x23};

fn flatten_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    return tensor.flatten(2); // equivalent to tensor.reshape(4,2)
}
>>> [[0,1],[2,3],[4,5],[6,7]]
```
