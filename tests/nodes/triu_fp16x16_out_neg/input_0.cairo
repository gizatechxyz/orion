use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1376256, sign: false });
    data.append(FP16x16 { mag: 7471104, sign: true });
    data.append(FP16x16 { mag: 5767168, sign: true });
    data.append(FP16x16 { mag: 7274496, sign: true });
    data.append(FP16x16 { mag: 131072, sign: true });
    data.append(FP16x16 { mag: 6029312, sign: true });
    data.append(FP16x16 { mag: 7077888, sign: true });
    data.append(FP16x16 { mag: 7208960, sign: false });
    data.append(FP16x16 { mag: 4063232, sign: false });
    data.append(FP16x16 { mag: 8192000, sign: false });
    data.append(FP16x16 { mag: 1703936, sign: true });
    data.append(FP16x16 { mag: 6225920, sign: true });
    data.append(FP16x16 { mag: 1114112, sign: false });
    data.append(FP16x16 { mag: 4980736, sign: true });
    data.append(FP16x16 { mag: 1507328, sign: false });
    data.append(FP16x16 { mag: 2949120, sign: false });
    data.append(FP16x16 { mag: 3866624, sign: false });
    data.append(FP16x16 { mag: 6553600, sign: true });
    data.append(FP16x16 { mag: 917504, sign: true });
    data.append(FP16x16 { mag: 6356992, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
