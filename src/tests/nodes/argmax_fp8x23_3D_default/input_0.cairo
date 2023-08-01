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
    data.append(FixedType { mag: 1031798784, sign: false });
    data.append(FixedType { mag: 687865856, sign: true });
    data.append(FixedType { mag: 612368384, sign: false });
    data.append(FixedType { mag: 822083584, sign: true });
    data.append(FixedType { mag: 746586112, sign: false });
    data.append(FixedType { mag: 226492416, sign: false });
    data.append(FixedType { mag: 268435456, sign: false });
    data.append(FixedType { mag: 553648128, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}