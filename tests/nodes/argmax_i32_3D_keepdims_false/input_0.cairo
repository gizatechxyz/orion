use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 17, sign: true });
    data.append(i32 { mag: 70, sign: false });
    data.append(i32 { mag: 84, sign: false });
    data.append(i32 { mag: 85, sign: true });
    data.append(i32 { mag: 97, sign: false });
    data.append(i32 { mag: 18, sign: false });
    data.append(i32 { mag: 115, sign: true });
    data.append(i32 { mag: 124, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
