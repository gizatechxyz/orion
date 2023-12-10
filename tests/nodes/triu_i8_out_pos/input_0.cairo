use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 27, sign: false });
    data.append(i8 { mag: 17, sign: false });
    data.append(i8 { mag: 50, sign: false });
    data.append(i8 { mag: 2, sign: true });
    data.append(i8 { mag: 111, sign: true });
    data.append(i8 { mag: 31, sign: true });
    data.append(i8 { mag: 41, sign: false });
    data.append(i8 { mag: 51, sign: true });
    data.append(i8 { mag: 60, sign: false });
    data.append(i8 { mag: 65, sign: true });
    data.append(i8 { mag: 106, sign: true });
    data.append(i8 { mag: 54, sign: true });
    data.append(i8 { mag: 103, sign: false });
    data.append(i8 { mag: 85, sign: false });
    data.append(i8 { mag: 66, sign: true });
    data.append(i8 { mag: 32, sign: false });
    data.append(i8 { mag: 106, sign: false });
    data.append(i8 { mag: 23, sign: true });
    data.append(i8 { mag: 59, sign: true });
    data.append(i8 { mag: 66, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
