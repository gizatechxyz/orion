use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 4063232, sign: true });
    data.append(FP16x16 { mag: 6946816, sign: false });
    data.append(FP16x16 { mag: 3407872, sign: true });
    data.append(FP16x16 { mag: 7340032, sign: true });
    data.append(FP16x16 { mag: 7667712, sign: true });
    data.append(FP16x16 { mag: 7536640, sign: false });
    data.append(FP16x16 { mag: 7012352, sign: false });
    data.append(FP16x16 { mag: 6750208, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
