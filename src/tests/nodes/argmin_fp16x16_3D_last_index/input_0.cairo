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
    data.append(FixedType { mag: 5832704, sign: false });
    data.append(FixedType { mag: 3145728, sign: false });
    data.append(FixedType { mag: 7929856, sign: true });
    data.append(FixedType { mag: 131072, sign: true });
    data.append(FixedType { mag: 5701632, sign: true });
    data.append(FixedType { mag: 2359296, sign: false });
    data.append(FixedType { mag: 1572864, sign: true });
    data.append(FixedType { mag: 7602176, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}