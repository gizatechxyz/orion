# tensor.squeeze

```rust 
   fn squeeze(self: @Tensor<T>, axes: Option<Span<i32>>) -> Tensor<T>;
```

Removes dimensions of size 1 from the shape of a tensor.

## Args

* `self`(`@Tensor<T>`) - Tensor of data to calculate non-zero indices.  
* `axes`(`Option<Span<i32>>`) - List of integers indicating the dimensions to squeeze.  

## Returns 

A new `Tensor<T>` Reshaped tensor with same data as input.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn squeeze_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![1, 2, 1, 2, 1].span(), 
        data: array![1, 1, 1, 1].span(), 
    );

    return tensor.squeeze(axes: Option::None(());
}
>>> [[1 1]
     [1 1]]
```
