# tensor.clip

```rust 
   fn clip(self: @Tensor<T>, min: T, max: T) -> Tensor<T>;
```

Clip operator limits the given input within an interval.

## Args

* `self`(`@Tensor<T>`) - Input tensor whose elements to be clipped.
* `min`(`Option<T>`) - Minimum value, under which element is replaced by min.
* `max`(`Option<T>`) - Maximum value, above which element is replaced by max.

## Returns 

Output `Tensor<T>` with clipped input elements.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn clip_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![2, 3].span(), 
        data: array![[ 1, 2, 3],[4, 5, 6]].span(), 
    );

    return tensor.clip(
        min: Option::None(()), 
        max: Option::Some(3),
    );
}
>>> [[1. 2. 3.]
     [3. 3. 3.]]
```
