use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 46451, sign: true });
    data.append(FP16x16 { mag: 3984, sign: true });
    data.append(FP16x16 { mag: 82191, sign: true });
    data.append(FP16x16 { mag: 9056, sign: true });
    data.append(FP16x16 { mag: 104929, sign: false });
    data.append(FP16x16 { mag: 9318, sign: true });
    data.append(FP16x16 { mag: 169836, sign: false });
    data.append(FP16x16 { mag: 106443, sign: false });
    data.append(FP16x16 { mag: 143696, sign: false });
    data.append(FP16x16 { mag: 99052, sign: false });
    data.append(FP16x16 { mag: 176999, sign: false });
    data.append(FP16x16 { mag: 54192, sign: false });
    data.append(FP16x16 { mag: 34801, sign: false });
    data.append(FP16x16 { mag: 1418, sign: false });
    data.append(FP16x16 { mag: 54547, sign: false });
    data.append(FP16x16 { mag: 80734, sign: false });
    data.append(FP16x16 { mag: 21511, sign: true });
    data.append(FP16x16 { mag: 244326, sign: true });
    data.append(FP16x16 { mag: 16281, sign: true });
    data.append(FP16x16 { mag: 99990, sign: false });
    data.append(FP16x16 { mag: 10550, sign: true });
    data.append(FP16x16 { mag: 48534, sign: false });
    data.append(FP16x16 { mag: 28730, sign: false });
    data.append(FP16x16 { mag: 10662, sign: false });
    data.append(FP16x16 { mag: 86081, sign: true });
    data.append(FP16x16 { mag: 42891, sign: true });
    data.append(FP16x16 { mag: 120456, sign: true });
    data.append(FP16x16 { mag: 163181, sign: false });
    data.append(FP16x16 { mag: 58800, sign: false });
    data.append(FP16x16 { mag: 5886, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
