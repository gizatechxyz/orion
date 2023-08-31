use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
use orion::numbers::FP8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::Tensor_fp8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 4332983, sign: true });
    data.append(FP8x23 { mag: 5907229, sign: true });
    data.append(FP8x23 { mag: 7324851, sign: false });
    data.append(FP8x23 { mag: 1964182, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}