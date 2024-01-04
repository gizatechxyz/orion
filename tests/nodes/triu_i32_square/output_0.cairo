use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 59, sign: true });
    data.append(i32 { mag: 103, sign: false });
    data.append(i32 { mag: 85, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 60, sign: true });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 113, sign: true });
    data.append(i32 { mag: 40, sign: false });
    data.append(i32 { mag: 66, sign: false });
    data.append(i32 { mag: 53, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 13, sign: false });
    data.append(i32 { mag: 30, sign: true });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
