use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 67, sign: true });
    data.append(i8 { mag: 99, sign: true });
    data.append(i8 { mag: 95, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 13, sign: true });
    data.append(i8 { mag: 113, sign: true });
    data.append(i8 { mag: 68, sign: true });
    data.append(i8 { mag: 99, sign: false });
    data.append(i8 { mag: 79, sign: false });
    data.append(i8 { mag: 20, sign: false });
    data.append(i8 { mag: 88, sign: false });
    data.append(i8 { mag: 120, sign: false });
    data.append(i8 { mag: 65, sign: true });
    data.append(i8 { mag: 112, sign: true });
    data.append(i8 { mag: 104, sign: true });
    data.append(i8 { mag: 106, sign: true });
    data.append(i8 { mag: 117, sign: true });
    data.append(i8 { mag: 8, sign: true });
    data.append(i8 { mag: 17, sign: true });
    data.append(i8 { mag: 65, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
