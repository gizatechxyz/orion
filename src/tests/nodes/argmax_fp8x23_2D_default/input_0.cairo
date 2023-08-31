use array::ArrayTrait;

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
use orion::operators::tensor::implementations::tensor_fp8x23::Tensor_fp8x23;use orion::numbers::FP8x23;


fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 142606336, sign: false });
    data.append(FP8x23 { mag: 411041792, sign: false });
    data.append(FP8x23 { mag: 956301312, sign: false });
    data.append(FP8x23 { mag: 385875968, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}