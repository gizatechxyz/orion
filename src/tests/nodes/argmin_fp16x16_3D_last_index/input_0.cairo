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
    data.append(FixedType { mag: 6553600, sign: false });
    data.append(FixedType { mag: 7667712, sign: true });
    data.append(FixedType { mag: 6815744, sign: true });
    data.append(FixedType { mag: 6488064, sign: true });
    data.append(FixedType { mag: 7274496, sign: false });
    data.append(FixedType { mag: 6750208, sign: false });
    data.append(FixedType { mag: 7405568, sign: false });
    data.append(FixedType { mag: 720896, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}