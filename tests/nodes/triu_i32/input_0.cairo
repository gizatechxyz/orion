use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 43, sign: false });
    data.append(i32 { mag: 58, sign: true });
    data.append(i32 { mag: 22, sign: true });
    data.append(i32 { mag: 28, sign: true });
    data.append(i32 { mag: 93, sign: true });
    data.append(i32 { mag: 60, sign: false });
    data.append(i32 { mag: 72, sign: false });
    data.append(i32 { mag: 86, sign: false });
    data.append(i32 { mag: 95, sign: false });
    data.append(i32 { mag: 45, sign: false });
    data.append(i32 { mag: 98, sign: false });
    data.append(i32 { mag: 54, sign: false });
    data.append(i32 { mag: 119, sign: true });
    data.append(i32 { mag: 10, sign: false });
    data.append(i32 { mag: 17, sign: false });
    data.append(i32 { mag: 91, sign: true });
    data.append(i32 { mag: 90, sign: true });
    data.append(i32 { mag: 112, sign: false });
    data.append(i32 { mag: 55, sign: false });
    data.append(i32 { mag: 56, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
