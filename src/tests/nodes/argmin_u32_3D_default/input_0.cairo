use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::FixedImpl;
use orion::operators::tensor::implementations::tensor_u32_fp16x16::Tensor_u32_fp16x16;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(137);
    data.append(2);
    data.append(134);
    data.append(142);
    data.append(236);
    data.append(246);
    data.append(96);
    data.append(150);

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}