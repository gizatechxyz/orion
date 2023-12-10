use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 73, sign: false });
    data.append(i8 { mag: 34, sign: false });
    data.append(i8 { mag: 40, sign: true });
    data.append(i8 { mag: 19, sign: false });
    data.append(i8 { mag: 52, sign: false });
    data.append(i8 { mag: 14, sign: true });
    data.append(i8 { mag: 15, sign: true });
    data.append(i8 { mag: 122, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
