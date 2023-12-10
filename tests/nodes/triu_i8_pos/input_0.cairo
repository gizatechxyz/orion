use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 12, sign: true });
    data.append(i8 { mag: 49, sign: false });
    data.append(i8 { mag: 80, sign: false });
    data.append(i8 { mag: 66, sign: false });
    data.append(i8 { mag: 94, sign: false });
    data.append(i8 { mag: 50, sign: true });
    data.append(i8 { mag: 71, sign: true });
    data.append(i8 { mag: 30, sign: false });
    data.append(i8 { mag: 97, sign: true });
    data.append(i8 { mag: 50, sign: false });
    data.append(i8 { mag: 11, sign: true });
    data.append(i8 { mag: 73, sign: true });
    data.append(i8 { mag: 79, sign: true });
    data.append(i8 { mag: 76, sign: true });
    data.append(i8 { mag: 69, sign: false });
    data.append(i8 { mag: 21, sign: false });
    data.append(i8 { mag: 119, sign: true });
    data.append(i8 { mag: 124, sign: true });
    data.append(i8 { mag: 81, sign: true });
    data.append(i8 { mag: 114, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
