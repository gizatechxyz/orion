## tensor.reduce_log_sum

```rust 
   fn reduce_log_sum(self: @Tensor<T>, axis: usize, keepdims: bool) -> Tensor<T>;
```

Computes the log sum of the input tensor's elements along the provided axes.
## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `axis`(`usize`) - The dimension to reduce.
* `keepdims`(`bool`) - If true, retains reduced dimensions with length 1.

## Panics 

* Panics if axis is not in the range of the input tensor's dimensions.

## Returns

A new `Tensor<T>` instance with the specified axis reduced by summing its elements.
## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor};
use orion::numbers::{FixedTrait, FP16x16};

fn reduce_log_sum() -> Tensor<FP16x16> {

   let mut sizes = ArrayTrait::new();
   sizes.append(2);
   sizes.append(2);
   sizes.append(2);

   let mut data = ArrayTrait::new();
   data.append(FixedTrait::new_unscaled(1, false));
   data.append(FixedTrait::new_unscaled(2, false));
   data.append(FixedTrait::new_unscaled(3, false));
   data.append(FixedTrait::new_unscaled(4, false));
   data.append(FixedTrait::new_unscaled(5, false));
   data.append(FixedTrait::new_unscaled(6, false));
   data.append(FixedTrait::new_unscaled(7, false));
   data.append(FixedTrait::new_unscaled(8, false));

   let tensor = TensorTrait::<FP16x16>::new(sizes.span(), data.span());

    We can call `reduce_log_sum` function as follows.
    return tensor.reduce_log_sum(axis: 2, keepdims: false);
}
>>> [[0x11938, 0x1f203], [0x265d9, 0x2b540]]
```
