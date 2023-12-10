use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 462, sign: false });
    data.append(i32 { mag: 525, sign: false });
    data.append(i32 { mag: 594, sign: false });
    data.append(i32 { mag: 669, sign: false });
    data.append(i32 { mag: 750, sign: false });
    data.append(i32 { mag: 837, sign: false });
    data.append(i32 { mag: 930, sign: false });
    data.append(i32 { mag: 1029, sign: false });
    data.append(i32 { mag: 1134, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
