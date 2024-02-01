# tensor.range

```rust 
   fn range(start: T, end: T, step: T) -> Tensor<T>;
```

Generate a tensor containing a sequence of numbers that begin at start and extends by increments of delta up to limit (exclusive).


* `start`(`T`) - First entry for the range of output values.
* `end`(`T`) - Exclusive upper limit for the range of output values.
* `step `(`T`) - Value to step by.

## Returns

A 1-D tensor with same type as the inputs containing generated range of values.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::I32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::numbers::NumberTrait;


fn range_example() -> Tensor<i32> {
    return TensorTrait::range(21,2,-3);
}
>>> [21 18 15 12  9  6  3]
```
