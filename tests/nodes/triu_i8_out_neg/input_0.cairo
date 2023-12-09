use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 29, sign: false });
    data.append(i8 { mag: 63, sign: true });
    data.append(i8 { mag: 73, sign: true });
    data.append(i8 { mag: 20, sign: true });
    data.append(i8 { mag: 18, sign: true });
    data.append(i8 { mag: 18, sign: true });
    data.append(i8 { mag: 28, sign: false });
    data.append(i8 { mag: 56, sign: false });
    data.append(i8 { mag: 114, sign: true });
    data.append(i8 { mag: 49, sign: true });
    data.append(i8 { mag: 97, sign: true });
    data.append(i8 { mag: 125, sign: true });
    data.append(i8 { mag: 102, sign: true });
    data.append(i8 { mag: 7, sign: false });
    data.append(i8 { mag: 89, sign: false });
    data.append(i8 { mag: 21, sign: true });
    data.append(i8 { mag: 84, sign: false });
    data.append(i8 { mag: 58, sign: true });
    data.append(i8 { mag: 12, sign: false });
    data.append(i8 { mag: 93, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
