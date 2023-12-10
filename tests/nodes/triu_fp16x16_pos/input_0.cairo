use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2686976, sign: false });
    data.append(FP16x16 { mag: 1703936, sign: true });
    data.append(FP16x16 { mag: 5570560, sign: true });
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 6815744, sign: true });
    data.append(FP16x16 { mag: 4521984, sign: true });
    data.append(FP16x16 { mag: 2228224, sign: true });
    data.append(FP16x16 { mag: 6553600, sign: true });
    data.append(FP16x16 { mag: 8126464, sign: true });
    data.append(FP16x16 { mag: 6356992, sign: false });
    data.append(FP16x16 { mag: 2031616, sign: false });
    data.append(FP16x16 { mag: 4128768, sign: false });
    data.append(FP16x16 { mag: 2883584, sign: true });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 7667712, sign: false });
    data.append(FP16x16 { mag: 4980736, sign: false });
    data.append(FP16x16 { mag: 3145728, sign: true });
    data.append(FP16x16 { mag: 3538944, sign: true });
    data.append(FP16x16 { mag: 5570560, sign: false });
    data.append(FP16x16 { mag: 3342336, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
