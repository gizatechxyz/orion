# tensor.binarizer

```rust
 fn binarizer(self: @Tensor<T>, threshold: Option<T>) -> Tensor<T>
```

Maps the values of a tensor element-wise to 0 or 1 based on the comparison against a threshold value.

## Args
* `self`(`@Tensor<T>`) - The input tensor to be binarized.
* `threshold`(`Option<T>`) - The threshold for the binarization operation.

## Returns
A new `Tensor<T>` of the same shape as the input tensor with binarized values.

## Type Constraints

Constrain input and output types to fixed point numbers.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FixedTrait, FP8x23};

fn binarizer_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![2, 2].span(),
        data: array![
            FixedTrait::new(0, false),
            FixedTrait::new(1, false),
            FixedTrait::new(2, false),
            FixedTrait::new(3, false)
        ]
            .span(),
    );
    let threshold = Option::Some(FixedTrait::new(1, false))

    return tensor.binarizer(@tensor, threshold);
}
>>> [0, 0, 8388608, 8388608]
    // The fixed point representation of
    [0, 0, 1, 1]
```
