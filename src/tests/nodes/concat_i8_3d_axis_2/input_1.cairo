use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
use orion::operators::tensor::implementations::tensor_fp16x16::Tensor_fp16x16;
use orion::numbers::FP16x16;

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 27, sign: false });
    data.append(FP16x16 { mag: 28, sign: false });
    data.append(FP16x16 { mag: 29, sign: false });
    data.append(FP16x16 { mag: 30, sign: false });
    data.append(FP16x16 { mag: 31, sign: false });
    data.append(FP16x16 { mag: 32, sign: false });
    data.append(FP16x16 { mag: 33, sign: false });
    data.append(FP16x16 { mag: 34, sign: false });
    data.append(FP16x16 { mag: 35, sign: false });
    data.append(FP16x16 { mag: 36, sign: false });
    data.append(FP16x16 { mag: 37, sign: false });
    data.append(FP16x16 { mag: 38, sign: false });
    data.append(FP16x16 { mag: 39, sign: false });
    data.append(FP16x16 { mag: 40, sign: false });
    data.append(FP16x16 { mag: 41, sign: false });
    data.append(FP16x16 { mag: 42, sign: false });
    data.append(FP16x16 { mag: 43, sign: false });
    data.append(FP16x16 { mag: 44, sign: false });
    data.append(FP16x16 { mag: 45, sign: false });
    data.append(FP16x16 { mag: 46, sign: false });
    data.append(FP16x16 { mag: 47, sign: false });
    data.append(FP16x16 { mag: 48, sign: false });
    data.append(FP16x16 { mag: 49, sign: false });
    data.append(FP16x16 { mag: 50, sign: false });
    data.append(FP16x16 { mag: 51, sign: false });
    data.append(FP16x16 { mag: 52, sign: false });
    data.append(FP16x16 { mag: 53, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}