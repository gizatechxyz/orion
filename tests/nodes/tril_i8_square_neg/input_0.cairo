use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 8, sign: true });
    data.append(i8 { mag: 30, sign: true });
    data.append(i8 { mag: 44, sign: false });
    data.append(i8 { mag: 25, sign: true });
    data.append(i8 { mag: 42, sign: false });
    data.append(i8 { mag: 34, sign: true });
    data.append(i8 { mag: 68, sign: true });
    data.append(i8 { mag: 18, sign: true });
    data.append(i8 { mag: 105, sign: false });
    data.append(i8 { mag: 56, sign: true });
    data.append(i8 { mag: 97, sign: false });
    data.append(i8 { mag: 92, sign: false });
    data.append(i8 { mag: 11, sign: true });
    data.append(i8 { mag: 117, sign: false });
    data.append(i8 { mag: 35, sign: false });
    data.append(i8 { mag: 72, sign: true });
    data.append(i8 { mag: 103, sign: false });
    data.append(i8 { mag: 73, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
