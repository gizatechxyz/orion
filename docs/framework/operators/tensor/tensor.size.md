# tensor.size

```rust 
   fn size(self: @Tensor<T>) -> Tensor<T>;
```

Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - An input tensor.

## Returns 

A new `Tensor<T>` of Total number of elements of the input tensor.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};

fn size_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![2, 3].span(), 
        data: array![[[1, 2, 3], [4, 5, 6]]].span(), 
    );

    return tensor.size();
}
>>> [6]
```
