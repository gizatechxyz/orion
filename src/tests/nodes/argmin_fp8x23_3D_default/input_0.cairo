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
    data.append(FixedType { mag: 134217728, sign: true });
    data.append(FixedType { mag: 511705088, sign: true });
    data.append(FixedType { mag: 637534208, sign: false });
    data.append(FixedType { mag: 167772160, sign: true });
    data.append(FixedType { mag: 746586112, sign: false });
    data.append(FixedType { mag: 293601280, sign: true });
    data.append(FixedType { mag: 461373440, sign: true });
    data.append(FixedType { mag: 402653184, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}