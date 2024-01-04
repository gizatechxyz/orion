use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 33, sign: true });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 44, sign: false });
    data.append(i32 { mag: 40, sign: true });
    data.append(i32 { mag: 14, sign: false });
    data.append(i32 { mag: 8, sign: true });
    data.append(i32 { mag: 90, sign: false });
    data.append(i32 { mag: 37, sign: true });
    data.append(i32 { mag: 73, sign: false });
    data.append(i32 { mag: 61, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 32, sign: true });
    data.append(i32 { mag: 86, sign: false });
    data.append(i32 { mag: 55, sign: false });
    data.append(i32 { mag: 106, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 59, sign: true });
    data.append(i32 { mag: 94, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
