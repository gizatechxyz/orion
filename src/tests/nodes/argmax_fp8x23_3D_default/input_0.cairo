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
    data.append(FixedType { mag: 914358272, sign: true });
    data.append(FixedType { mag: 956301312, sign: true });
    data.append(FixedType { mag: 268435456, sign: true });
    data.append(FixedType { mag: 981467136, sign: true });
    data.append(FixedType { mag: 687865856, sign: true });
    data.append(FixedType { mag: 511705088, sign: false });
    data.append(FixedType { mag: 58720256, sign: false });
    data.append(FixedType { mag: 209715200, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}