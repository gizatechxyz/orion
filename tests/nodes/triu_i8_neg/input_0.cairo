use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 4, sign: false });
    data.append(i8 { mag: 11, sign: true });
    data.append(i8 { mag: 70, sign: false });
    data.append(i8 { mag: 41, sign: true });
    data.append(i8 { mag: 81, sign: true });
    data.append(i8 { mag: 62, sign: true });
    data.append(i8 { mag: 89, sign: false });
    data.append(i8 { mag: 67, sign: true });
    data.append(i8 { mag: 69, sign: false });
    data.append(i8 { mag: 61, sign: true });
    data.append(i8 { mag: 34, sign: true });
    data.append(i8 { mag: 100, sign: false });
    data.append(i8 { mag: 93, sign: true });
    data.append(i8 { mag: 39, sign: false });
    data.append(i8 { mag: 19, sign: true });
    data.append(i8 { mag: 2, sign: false });
    data.append(i8 { mag: 43, sign: false });
    data.append(i8 { mag: 43, sign: false });
    data.append(i8 { mag: 30, sign: true });
    data.append(i8 { mag: 110, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
