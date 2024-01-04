## tensor.is_inf

```rust
   fn is_inf(self: @Tensor<T>, detect_negative: Option<u8>, detect_positive: Option<u8>) -> Tensor<bool>;
```

Maps infinity to true and other values to false.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `detect_negative`(`Option<u8>`) - Optional Whether map negative infinity to true. Default to 1 so that negative infinity induces true.
* `detect_positive`(`Option<u8>`) - Optional Whether map positive infinity to true. Default to 1 so that positive infinity induces true.


## Returns

A new `Tensor<bool>` instance with entries set to true iff the input tensors corresponding element was infinity.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};    
use orion::operators::tensor::{BoolTensor, TensorTrait, Tensor, U32Tensor};

fn is_inf_example() -> Tensor<bool> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![6].span(), data: array![1, 0, NumberTrait::INF(), 8, NumberTrait::INF(), NumberTrait::INF()].span(),
    );

    return tensor.is_inf(detect_negative: Option::None, detect_positive: Option::None);
}
>>> [false, false, true, false, true, true]
```
