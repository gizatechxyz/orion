use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(7);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 42416, sign: false });
    data.append(FP16x16 { mag: 877, sign: false });
    data.append(FP16x16 { mag: 21463, sign: false });
    data.append(FP16x16 { mag: 55531, sign: false });
    data.append(FP16x16 { mag: 62444, sign: false });
    data.append(FP16x16 { mag: 30762, sign: false });
    data.append(FP16x16 { mag: 15704, sign: false });
    data.append(FP16x16 { mag: 36007, sign: false });
    data.append(FP16x16 { mag: 18900, sign: false });
    data.append(FP16x16 { mag: 3784, sign: false });
    data.append(FP16x16 { mag: 356, sign: false });
    data.append(FP16x16 { mag: 51406, sign: false });
    data.append(FP16x16 { mag: 57856, sign: false });
    data.append(FP16x16 { mag: 27283, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
