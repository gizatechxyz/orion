use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 123, sign: true });
    data.append(i8 { mag: 5, sign: true });
    data.append(i8 { mag: 8, sign: false });
    data.append(i8 { mag: 10, sign: true });
    data.append(i8 { mag: 112, sign: false });
    data.append(i8 { mag: 67, sign: true });
    data.append(i8 { mag: 60, sign: true });
    data.append(i8 { mag: 16, sign: false });
    data.append(i8 { mag: 56, sign: true });
    data.append(i8 { mag: 12, sign: false });
    data.append(i8 { mag: 42, sign: false });
    data.append(i8 { mag: 88, sign: true });
    data.append(i8 { mag: 114, sign: false });
    data.append(i8 { mag: 27, sign: true });
    data.append(i8 { mag: 48, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
