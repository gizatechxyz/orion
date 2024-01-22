# tensor.array_feature_extractor

```rust
    fn array_feature_extractor(self: @Tensor<T>, indices: Tensor<usize>) -> Tensor<T>;
```

Selects elements of the input tensor based on the indices passed applied to the last tensor axis. 

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `indices`(`Tensor<usize>`) - Tensor of indices.

## Panics

* Panics if indices tensor is not 1-dimensional.

## Returns

A new `Tensor<T>` of the same shape as the input tensor with selected elements based on provided indices.

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor, U32Tensor};

fn array_feature_extractor_example() -> Tensor<i32> {
    let input_tensor = TensorTrait::new(
        shape: array![3, 4].span(),
        data: array![
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11
        ].span(),
    );
    
    let indices = TensorTrait::<u32>::new(
        shape: array![2].span(), data: array![1, 3].span(),
    );

    return tensor.array_feature_extractor(@input_tensor, @indices);
}
>>> [[1, 3]
     [5, 7]
     [9, 11]]
```
