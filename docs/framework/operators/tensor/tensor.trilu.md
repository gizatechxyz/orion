# tensor.trilu

```rust 
   fn trilu(self: @Tensor<T>, upper: bool, k: i64) -> Tensor<T>;
```

Returns a new tensor with the uppper/lower triangular part of the tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `upper`(`bool`) - if true, returns the upper triangular part of the tensor, otherwise returns the lower part.
* `k`(`i64`) - value corresponding to the number diagonals above or below the main diagonal to exclude or include.

## Panics

* Panics if the dimension of the tensor is less than 2.

## Returns

A `Tensor<T>` instance with the uppper/lower triangular part of the tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn trilu_tensor_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 3, 3].span(), data: array![0, 4, 3, 2, 0, 9, 8, 2, 5, 2, 7, 2, 2, 6, 0, 2, 6 ,5].span(),
    );

    // We can call `trilu` function as follows.
    return tensor.trilu(false, 0);
}
>>> [[[0, 0, 0],[2, 0, 0], [8, 2, 5]], [[2, 0, 0], [2, 6, 0], [2, 6, 5]]]
```
