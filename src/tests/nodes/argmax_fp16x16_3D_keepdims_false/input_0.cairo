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
    data.append(FixedType { mag: 4128768, sign: false });
    data.append(FixedType { mag: 4390912, sign: true });
    data.append(FixedType { mag: 7274496, sign: true });
    data.append(FixedType { mag: 5111808, sign: true });
    data.append(FixedType { mag: 6684672, sign: true });
    data.append(FixedType { mag: 2818048, sign: true });
    data.append(FixedType { mag: 6553600, sign: true });
    data.append(FixedType { mag: 3407872, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}