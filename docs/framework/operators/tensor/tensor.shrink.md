# tensor.shrink

```rust
 fn shrink(self: @Tensor<T>, bias: Option<T>, lambd: Option<T>) -> Tensor<T>
```

Shrinks the input tensor element-wise to the output tensor with the same datatype and shape based on the following formula:
If x < -lambd: y = x + bias; If x > lambd: y = x - bias; Otherwise: y = 0.

## Args
* `self`(`@Tensor<T>`) - The input tensor to be shrinked.
* `bias`(`Option<T>`) - The bias value added to or subtracted from input tensor values.
* `lambd`(`Option<T>`) - The lambd value defining the shrink condition.

## Returns
A new `Tensor<T>` of the same datatype and shape as the input tensor with shrinked values.

## Type Constraints

Constrain input and output types to fixed point numbers.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FixedTrait, FP8x23};

fn shrink_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![2, 2].span(),
        data: array![
            FixedTrait::new(2, true),
            FixedTrait::new(1, true),
            FixedTrait::new(1, false),
            FixedTrait::new(2, false)
        ]
            .span(),
    );
    let bias = Option::Some(FixedTrait::new(1, false))
    let lambd = Option::Some(FixedTrait::new(1, false))

    return tensor.shrink(tensor, bias, lambd);
}
>>> [-8388608, 0, 0, 8388608]
    // The fixed point representation of
    [-1, 0, 0, 1]
```
