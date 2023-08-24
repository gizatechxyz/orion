use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn input_0() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FixedType { mag: 897581056, sign: true });
    data.append(FixedType { mag: 914358272, sign: true });
    data.append(FixedType { mag: 394264576, sign: true });
    data.append(FixedType { mag: 511705088, sign: true });
    data.append(FixedType { mag: 822083584, sign: true });
    data.append(FixedType { mag: 746586112, sign: false });
    data.append(FixedType { mag: 855638016, sign: true });
    data.append(FixedType { mag: 226492416, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}