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
    data.append(FixedType { mag: 4653056, sign: false });
    data.append(FixedType { mag: 6750208, sign: true });
    data.append(FixedType { mag: 393216, sign: true });
    data.append(FixedType { mag: 6160384, sign: true });
    data.append(FixedType { mag: 4063232, sign: true });
    data.append(FixedType { mag: 3932160, sign: true });
    data.append(FixedType { mag: 8257536, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}