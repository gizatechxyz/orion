use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 87783, sign: false });
    data.append(FP16x16 { mag: 105727, sign: false });
    data.append(FP16x16 { mag: 98446, sign: false });
    data.append(FP16x16 { mag: 113342, sign: false });
    data.append(FP16x16 { mag: 53524, sign: false });
    data.append(FP16x16 { mag: 83190, sign: false });
    data.append(FP16x16 { mag: 41921, sign: false });
    data.append(FP16x16 { mag: 72526, sign: false });
    data.append(FP16x16 { mag: 78945, sign: false });
    data.append(FP16x16 { mag: 73149, sign: false });
    data.append(FP16x16 { mag: 75553, sign: false });
    data.append(FP16x16 { mag: 98680, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
