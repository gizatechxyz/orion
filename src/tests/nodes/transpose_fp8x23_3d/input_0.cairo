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
    data.append(FixedType { mag: 50331648, sign: false });
    data.append(FixedType { mag: 226492416, sign: false });
    data.append(FixedType { mag: 830472192, sign: false });
    data.append(FixedType { mag: 343932928, sign: true });
    data.append(FixedType { mag: 1048576000, sign: true });
    data.append(FixedType { mag: 192937984, sign: false });
    data.append(FixedType { mag: 947912704, sign: true });
    data.append(FixedType { mag: 8388608, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}