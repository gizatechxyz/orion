use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 21, sign: false });
    data.append(i32 { mag: 13, sign: false });
    data.append(i32 { mag: 38, sign: false });
    data.append(i32 { mag: 47, sign: false });
    data.append(i32 { mag: 64, sign: false });
    data.append(i32 { mag: 57, sign: false });
    data.append(i32 { mag: 107, sign: false });
    data.append(i32 { mag: 104, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
