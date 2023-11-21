use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 6750208, sign: false });
    data.append(FP16x16 { mag: 6225920, sign: false });
    data.append(FP16x16 { mag: 2031616, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 4980736, sign: false });
    data.append(FP16x16 { mag: 2293760, sign: false });
    data.append(FP16x16 { mag: 7864320, sign: false });
    data.append(FP16x16 { mag: 3276800, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 2818048, sign: true });
    data.append(FP16x16 { mag: 2162688, sign: true });
    data.append(FP16x16 { mag: 7077888, sign: true });
    data.append(FP16x16 { mag: 2555904, sign: true });
    data.append(FP16x16 { mag: 2686976, sign: false });
    data.append(FP16x16 { mag: 2818048, sign: false });
    data.append(FP16x16 { mag: 3604480, sign: false });
    data.append(FP16x16 { mag: 5963776, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
