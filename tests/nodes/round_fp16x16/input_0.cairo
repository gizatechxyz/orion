use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(15);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 32768, sign: false });
    data.append(FP16x16 { mag: 58982, sign: false });
    data.append(FP16x16 { mag: 78643, sign: false });
    data.append(FP16x16 { mag: 98304, sign: false });
    data.append(FP16x16 { mag: 117964, sign: false });
    data.append(FP16x16 { mag: 150732, sign: false });
    data.append(FP16x16 { mag: 163840, sign: false });
    data.append(FP16x16 { mag: 176947, sign: false });
    data.append(FP16x16 { mag: 72089, sign: true });
    data.append(FP16x16 { mag: 98304, sign: true });
    data.append(FP16x16 { mag: 124518, sign: true });
    data.append(FP16x16 { mag: 144179, sign: true });
    data.append(FP16x16 { mag: 163840, sign: true });
    data.append(FP16x16 { mag: 183500, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
