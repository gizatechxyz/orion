use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(4);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 81920, sign: false });
    data.append(FP16x16 { mag: 114688, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 98304, sign: false });
    data.append(FP16x16 { mag: 114688, sign: false });
    data.append(FP16x16 { mag: 147456, sign: false });
    data.append(FP16x16 { mag: 163840, sign: false });
    data.append(FP16x16 { mag: 163840, sign: false });
    data.append(FP16x16 { mag: 180224, sign: false });
    data.append(FP16x16 { mag: 212992, sign: false });
    data.append(FP16x16 { mag: 229376, sign: false });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 212992, sign: false });
    data.append(FP16x16 { mag: 245760, sign: false });
    data.append(FP16x16 { mag: 262144, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
