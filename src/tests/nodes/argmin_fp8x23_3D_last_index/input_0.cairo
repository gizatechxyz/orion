use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn input_0() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedType { mag: 603979776, sign: false });
    data.append(FixedType { mag: 671088640, sign: true });
    data.append(FixedType { mag: 822083584, sign: false });
    data.append(FixedType { mag: 746586112, sign: true });
    data.append(FixedType { mag: 377487360, sign: false });
    data.append(FixedType { mag: 293601280, sign: false });
    data.append(FixedType { mag: 645922816, sign: false });
    data.append(FixedType { mag: 880803840, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}