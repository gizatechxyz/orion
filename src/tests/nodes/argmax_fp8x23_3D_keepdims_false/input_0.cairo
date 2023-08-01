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
    data.append(FixedType { mag: 251658240, sign: true });
    data.append(FixedType { mag: 16777216, sign: false });
    data.append(FixedType { mag: 1006632960, sign: false });
    data.append(FixedType { mag: 184549376, sign: false });
    data.append(FixedType { mag: 570425344, sign: true });
    data.append(FixedType { mag: 822083584, sign: true });
    data.append(FixedType { mag: 629145600, sign: false });
    data.append(FixedType { mag: 947912704, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}