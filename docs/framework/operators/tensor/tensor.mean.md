# tensor.mean

```rust 
   fn mean(args: Span<Tensor<T>>) -> Tensor<T>;
```

Element-wise mean of each of the input tensors.


* `args`(`Span<Tensor<T>>`) - List of tensors for mean.

## Returns

Output tensor. 

## Examples

```rust
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::numbers::{FixedTrait, FP8x23};


fn example() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16777216, sign: true });
    data.append(FP8x23 { mag: 16777216, sign: false });
    data.append(FP8x23 { mag: 16777216, sign: true });
    data.append(FP8x23 { mag: 16777216, sign: false });
    let tensor1 = TensorTrait::new(shape.span(), data.span());

    let mut shape2 = ArrayTrait::<usize>::new();
    shape2.append(2);
    shape2.append(2);

    let mut data2 = ArrayTrait::new();
    data2.append(FP8x23 { mag: 8388608, sign: false });
    data2.append(FP8x23 { mag: 0, sign: false });
    data2.append(FP8x23 { mag: 0, sign: false });
    data2.append(FP8x23 { mag: 8388608, sign: false });
    let tensor2 = TensorTrait::new(shape2.span(), data2.span());
    return TensorTrait::mean(array![tensor1, tensor2].span());
}
>>> [FP8x23 { mag: 4194304, sign: false }, FP8x23 { mag: 8388608, sign: true }, FP8x23 { mag: 8388608, sign: false }, FP8x23 { mag: 12582912, sign: true }]
```
