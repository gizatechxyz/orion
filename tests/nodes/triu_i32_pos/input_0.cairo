use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 15, sign: true });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 49, sign: true });
    data.append(i32 { mag: 5, sign: false });
    data.append(i32 { mag: 117, sign: true });
    data.append(i32 { mag: 116, sign: false });
    data.append(i32 { mag: 43, sign: true });
    data.append(i32 { mag: 48, sign: false });
    data.append(i32 { mag: 49, sign: true });
    data.append(i32 { mag: 35, sign: false });
    data.append(i32 { mag: 98, sign: false });
    data.append(i32 { mag: 40, sign: false });
    data.append(i32 { mag: 35, sign: false });
    data.append(i32 { mag: 122, sign: true });
    data.append(i32 { mag: 8, sign: false });
    data.append(i32 { mag: 125, sign: false });
    data.append(i32 { mag: 102, sign: false });
    data.append(i32 { mag: 55, sign: true });
    data.append(i32 { mag: 39, sign: false });
    data.append(i32 { mag: 98, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
