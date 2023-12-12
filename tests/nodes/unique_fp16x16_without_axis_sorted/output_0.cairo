use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(27);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2490, sign: false });
    data.append(FP16x16 { mag: 10368, sign: false });
    data.append(FP16x16 { mag: 12496, sign: false });
    data.append(FP16x16 { mag: 36320, sign: false });
    data.append(FP16x16 { mag: 38624, sign: false });
    data.append(FP16x16 { mag: 48512, sign: false });
    data.append(FP16x16 { mag: 52000, sign: false });
    data.append(FP16x16 { mag: 55104, sign: false });
    data.append(FP16x16 { mag: 56992, sign: false });
    data.append(FP16x16 { mag: 71360, sign: false });
    data.append(FP16x16 { mag: 80256, sign: false });
    data.append(FP16x16 { mag: 82560, sign: false });
    data.append(FP16x16 { mag: 89920, sign: false });
    data.append(FP16x16 { mag: 90048, sign: false });
    data.append(FP16x16 { mag: 94592, sign: false });
    data.append(FP16x16 { mag: 108736, sign: false });
    data.append(FP16x16 { mag: 114048, sign: false });
    data.append(FP16x16 { mag: 120832, sign: false });
    data.append(FP16x16 { mag: 122432, sign: false });
    data.append(FP16x16 { mag: 129472, sign: false });
    data.append(FP16x16 { mag: 132224, sign: false });
    data.append(FP16x16 { mag: 147456, sign: false });
    data.append(FP16x16 { mag: 167808, sign: false });
    data.append(FP16x16 { mag: 171904, sign: false });
    data.append(FP16x16 { mag: 177152, sign: false });
    data.append(FP16x16 { mag: 182272, sign: false });
    data.append(FP16x16 { mag: 195712, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
