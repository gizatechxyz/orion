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
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 7340032, sign: false });
    data.append(FP16x16 { mag: 5701632, sign: true });
    data.append(FP16x16 { mag: 1376256, sign: true });
    data.append(FP16x16 { mag: 2424832, sign: false });
    data.append(FP16x16 { mag: 3866624, sign: true });
    data.append(FP16x16 { mag: 2359296, sign: true });
    data.append(FP16x16 { mag: 589824, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
