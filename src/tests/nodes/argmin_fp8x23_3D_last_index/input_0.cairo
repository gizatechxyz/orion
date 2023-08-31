use array::ArrayTrait;

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
use orion::operators::tensor::implementations::tensor_fp8x23::Tensor_fp8x23;use orion::numbers::FP8x23;


fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 603979776, sign: false });
    data.append(FP8x23 { mag: 671088640, sign: true });
    data.append(FP8x23 { mag: 822083584, sign: false });
    data.append(FP8x23 { mag: 746586112, sign: true });
    data.append(FP8x23 { mag: 377487360, sign: false });
    data.append(FP8x23 { mag: 293601280, sign: false });
    data.append(FP8x23 { mag: 645922816, sign: false });
    data.append(FP8x23 { mag: 880803840, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}