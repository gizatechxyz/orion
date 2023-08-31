use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
use orion::operators::tensor::implementations::tensor_fp16x16::Tensor_fp16x16;
use orion::numbers::FP16x16;

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 54, sign: false });
    data.append(FP16x16 { mag: 55, sign: false });
    data.append(FP16x16 { mag: 56, sign: false });
    data.append(FP16x16 { mag: 57, sign: false });
    data.append(FP16x16 { mag: 58, sign: false });
    data.append(FP16x16 { mag: 59, sign: false });
    data.append(FP16x16 { mag: 60, sign: false });
    data.append(FP16x16 { mag: 61, sign: false });
    data.append(FP16x16 { mag: 62, sign: false });
    data.append(FP16x16 { mag: 63, sign: false });
    data.append(FP16x16 { mag: 64, sign: false });
    data.append(FP16x16 { mag: 65, sign: false });
    data.append(FP16x16 { mag: 66, sign: false });
    data.append(FP16x16 { mag: 67, sign: false });
    data.append(FP16x16 { mag: 68, sign: false });
    data.append(FP16x16 { mag: 69, sign: false });
    data.append(FP16x16 { mag: 70, sign: false });
    data.append(FP16x16 { mag: 71, sign: false });
    data.append(FP16x16 { mag: 72, sign: false });
    data.append(FP16x16 { mag: 73, sign: false });
    data.append(FP16x16 { mag: 74, sign: false });
    data.append(FP16x16 { mag: 75, sign: false });
    data.append(FP16x16 { mag: 76, sign: false });
    data.append(FP16x16 { mag: 77, sign: false });
    data.append(FP16x16 { mag: 78, sign: false });
    data.append(FP16x16 { mag: 79, sign: false });
    data.append(FP16x16 { mag: 80, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}