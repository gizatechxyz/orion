#tensor.bitwise_not

```rust
    fn bitwise_not(self: @Tensor<T>) -> Tensor<usize>;
```

Computes the bitwise NOT of the tensor element-wise.

## Args

* `self`(`@Tensor<T>`) - The the tensor to be bitwise NOTed


## Returns

A new `Tensor<T>` with the same shape as the broadcasted input.

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn bitwise_not_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    );

    return tensor_1.bitwise_not();
}
>>> [4294967295,4294967294,4294967293,4294967292,4294967291,4294967290,4294967289,4294967288,4294967287]
```
