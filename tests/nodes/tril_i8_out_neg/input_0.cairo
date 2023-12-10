use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 13, sign: false });
    data.append(i8 { mag: 21, sign: false });
    data.append(i8 { mag: 12, sign: false });
    data.append(i8 { mag: 9, sign: false });
    data.append(i8 { mag: 49, sign: true });
    data.append(i8 { mag: 8, sign: true });
    data.append(i8 { mag: 100, sign: true });
    data.append(i8 { mag: 57, sign: true });
    data.append(i8 { mag: 94, sign: false });
    data.append(i8 { mag: 122, sign: false });
    data.append(i8 { mag: 106, sign: true });
    data.append(i8 { mag: 10, sign: true });
    data.append(i8 { mag: 57, sign: true });
    data.append(i8 { mag: 71, sign: false });
    data.append(i8 { mag: 121, sign: false });
    data.append(i8 { mag: 125, sign: false });
    data.append(i8 { mag: 31, sign: false });
    data.append(i8 { mag: 82, sign: false });
    data.append(i8 { mag: 57, sign: true });
    data.append(i8 { mag: 21, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
