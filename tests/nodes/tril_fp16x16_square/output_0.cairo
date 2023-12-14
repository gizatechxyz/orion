use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7864320, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 2293760, sign: true });
    data.append(FP16x16 { mag: 2097152, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 6225920, sign: true });
    data.append(FP16x16 { mag: 5701632, sign: true });
    data.append(FP16x16 { mag: 4653056, sign: true });
    data.append(FP16x16 { mag: 1638400, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 5373952, sign: true });
    data.append(FP16x16 { mag: 5242880, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 720896, sign: false });
    data.append(FP16x16 { mag: 7536640, sign: true });
    data.append(FP16x16 { mag: 5767168, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
