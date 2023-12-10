use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 116, sign: false });
    data.append(i8 { mag: 47, sign: false });
    data.append(i8 { mag: 24, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 61, sign: true });
    data.append(i8 { mag: 30, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 116, sign: false });
    data.append(i8 { mag: 55, sign: true });
    data.append(i8 { mag: 4, sign: true });
    data.append(i8 { mag: 115, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 100, sign: false });
    data.append(i8 { mag: 110, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 25, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
