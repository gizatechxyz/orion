## tensor.erf

```rust 
   fn erf(self: @Tensor<T>) -> Tensor<T>;
```

Computes the mean of the input tensor's elements along the provided axes.

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the the error function of the input tensor computed element-wise.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor};
use orion::numbers::{FixedTrait, FP16x16};

fn erf_example() -> Tensor<FP16x16> {
    // The erf inputs is [1.0, 0.134, 0.520, 2.0, 3.5, 5.164]
    let tensor = TensorTrait::<FP16x16>::new(
        shape: array![6].span(),
        data: array![
            FixedTrait::new_unscaled(65536, false),
            FixedTrait::new_unscaled(8832, false),
            FixedTrait::new_unscaled(34079, false),
            FixedTrait::new_unscaled(131072, false),
            FixedTrait::new_unscaled(229376, false),
            FixedTrait::new_unscaled(338428, false),
        ]
            .span(),
    );

    return tensor.erf();
}
>>> [55227,9560,35252,65229,65536,65536]
```
