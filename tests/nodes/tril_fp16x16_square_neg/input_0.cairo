use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 4390912, sign: false });
    data.append(FP16x16 { mag: 917504, sign: true });
    data.append(FP16x16 { mag: 6029312, sign: true });
    data.append(FP16x16 { mag: 5111808, sign: true });
    data.append(FP16x16 { mag: 7995392, sign: true });
    data.append(FP16x16 { mag: 2621440, sign: false });
    data.append(FP16x16 { mag: 3801088, sign: true });
    data.append(FP16x16 { mag: 6225920, sign: false });
    data.append(FP16x16 { mag: 6946816, sign: false });
    data.append(FP16x16 { mag: 3801088, sign: true });
    data.append(FP16x16 { mag: 3997696, sign: true });
    data.append(FP16x16 { mag: 5308416, sign: false });
    data.append(FP16x16 { mag: 4587520, sign: false });
    data.append(FP16x16 { mag: 3866624, sign: false });
    data.append(FP16x16 { mag: 4849664, sign: true });
    data.append(FP16x16 { mag: 3866624, sign: true });
    data.append(FP16x16 { mag: 6422528, sign: true });
    data.append(FP16x16 { mag: 7208960, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
