# TensorTrait::bit_shift

```rust
        fn bit_shift(tensor1: @Tensor<T>, tensor2: @Tensor<T>, direction: felt252) -> Tensor<T>;
```

Bitwise shift operator performs element-wise operation. For each input element, if the attribute "direction" is "RIGHT", this operator moves its binary representation toward the right side so that the input value is effectively decreased. If the attribute "direction" is "LEFT", bits of binary representation moves toward the left side, which results the increase of its actual value. 
The input tensor1 is the tensor to be shifted and another input tensor2 specifies the amounts of shifting. 

## Args

* `tensor1`(`@Tensor<T>`) - First operand, input to be shifted.
* `tensor2`(`@Tensor<T>`) - Second operand, amounts of shift.
* `direction`(@Tensor<T>) - Direction of moving bits. It can be either "RIGHT" (for right shift) or "LEFT" (for left shift).

## Returns

* Output tensor

## Examples

```rust

use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::U32TensorPartialEq;


fn example() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(21);
    data.append(7);
    data.append(18);
    data.append(43);
    data.append(49);
    data.append(49);
    data.append(4);
    data.append(28);
    data.append(24);
    let input_0 = TensorTrait::new(shape.span(), data.span());

    let mut shape1 = ArrayTrait::<usize>::new();
    shape1.append(3);
    shape1.append(3);

    let mut data1 = ArrayTrait::new();
    data1.append(4);
    data1.append(0);
    data1.append(0);
    data1.append(3);
    data1.append(1);
    data1.append(1);
    data1.append(1);
    data1.append(1);
    data1.append(0);
    let input_1 = TensorTrait::new(shape1.span(), data1.span());


    return TensorTrait::bit_shift(@input_0, @input_1, 'LEFT');
}
>>> [336 7 18 344 98 98 8 56 24]
```
