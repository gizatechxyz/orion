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
    data.append(FixedType { mag: 746586112, sign: true });
    data.append(FixedType { mag: 721420288, sign: true });
    data.append(FixedType { mag: 796917760, sign: false });
    data.append(FixedType { mag: 1040187392, sign: true });
    data.append(FixedType { mag: 109051904, sign: true });
    data.append(FixedType { mag: 1056964608, sign: false });
    data.append(FixedType { mag: 50331648, sign: false });
    data.append(FixedType { mag: 360710144, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}