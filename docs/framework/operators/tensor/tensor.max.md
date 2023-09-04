# tensor.max

```rust 
   fn max(self: @Tensor<T>) -> T;
```

Returns the maximum value in the tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

The maximum `T` value in the tensor.

Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn max_example() -> u32 {
    let tensor = TensorTrait::new(
        shape: array![2, 2, 2].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7].span(),
    );

    // We can call `max` function as follows.
    return tensor.max();
}
>>> 7
```
