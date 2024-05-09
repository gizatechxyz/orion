#tensor.modulo

```rust
    fn modulo(self: @Tensor<T>, divisor: @Tensor<T>, fmod: Option<bool>) -> Tensor<T>;
```

Computes element-wise modulo operation between two tensors with float modulo supported.

## Args

* `self` (`@Tensor<T>`): The dividend tensor.
* `other` (`@Tensor<T>`): The divisor tensor.
* `fmod` (`Option<bool>`): Optional attribute controlling the modulo behavior.

## Returns

A new `Tensor<T>` containing the element-wise modulo of the input tensors.

## Panics

* Panics if the shapes are not equal or broadcastable

## Examples:

Case 1: Integer modulo (default behavior)

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::numbers::NumberTrait;

fn mod_int_example() -> Tensor<i32> {
    let dividend = TensorTrait::<u32>::new(shape: array![2, 3].span(), data: array![5, 7, 2, 4, 10, 3].span());
    let divisor = TensorTrait::<u32>::new(shape: array![1,3].span(), data: array![2, 3, 4].span());

    return dividend.modulo(other: @divisor, fmod: Option::None(()));
}
>>> [[1, 1, 2], [0, 1, 3]]

Case 2: Float modulo

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn fmod_example() -> Tensor<FP16x16> {

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 466182, sign: false });
    data.append(FP16x16 { mag: 309008, sign: true });
    data.append(FP16x16 { mag: 529990, sign: true });
    data.append(FP16x16 { mag: 603028, sign: true });
    data.append(FP16x16 { mag: 607566, sign: true });
    data.append(FP16x16 { mag: 279123, sign: false });

    let dividend = TensorTrait::new(shape.span(), data.span())

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 182730, sign: false });
    data.append(FP16x16 { mag: 74626, sign: true });
    data.append(FP16x16 { mag: 423899, sign: false });

    let divisor = TensorTrait::new(shape.span(), data.span())

    return dividend.modulo(other: divisor, fmod: Option::Some(true));
}

>>> [[1.5369, -0.1603 , -1.6188], [-0.8368, -0.1611, 4.2591]]
```
