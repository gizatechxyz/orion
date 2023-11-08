# tensor.binarizer

```rust
 fn binarizer(self: @Tensor<T>, threshold: @T) -> Tensor<T>
```

Maps the values of a tensor element-wise to 0 or 1 based on the comparison against a threshold value.

## Args
* `self`(`@Tensor<T>`) - The input tensor to be binarized.
* `threshold`(`@T`) - The threshold for the binarization operation.

## Returns
A new `Tensor<T>` of the same shape as the input tensor with binarized values.

## Type Constraints

Constrain input and output types to fixed point and int32 tensors.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
use orion::numbers::{i32, IntegerTrait};

fn binarizer_example() -> Tensor<i32> {
    let tensor = TensorTrait::<i32>::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::new(1, true),
            IntegerTrait::new(0, false),
            IntegerTrait::new(1, false),
            IntegerTrait::new(2, false),
        ]
            .span(),
    );
    let threshold = IntegerTrait::<i32>::new(1, false)

    return tensor.binarizer(@tensor, @threshold);
}
>>> [0, 0, 0, 1]
```
