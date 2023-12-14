use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 35, sign: true });
    data.append(i8 { mag: 49, sign: true });
    data.append(i8 { mag: 45, sign: true });
    data.append(i8 { mag: 127, sign: true });
    data.append(i8 { mag: 50, sign: false });
    data.append(i8 { mag: 47, sign: true });
    data.append(i8 { mag: 24, sign: true });
    data.append(i8 { mag: 68, sign: false });
    data.append(i8 { mag: 31, sign: true });
    data.append(i8 { mag: 119, sign: true });
    data.append(i8 { mag: 59, sign: true });
    data.append(i8 { mag: 73, sign: false });
    data.append(i8 { mag: 28, sign: false });
    data.append(i8 { mag: 58, sign: true });
    data.append(i8 { mag: 7, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
