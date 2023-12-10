## tensor.is_nan

```rust
   fn is_nan(self: @Tensor<T>) -> Tensor<bool>;
```

Maps NaN to true and other values to false.

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

A new `Tensor<bool>` instance with entries set to true iff the input tensors corresponding element was NaN.

## Examples

use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{BoolTensor, TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FixedTrait, FP8x23};

fn is_nan_example() -> Tensor<bool> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 10066329, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FixedTrait::NaN());
    data.append(FP8x23 { mag: 23488102, sign: false });
    let tensor = TensorTrait::new(shape.span(), data.span())

    return tensor.is_nan();
}
>>> [false, false, true, false]
```
