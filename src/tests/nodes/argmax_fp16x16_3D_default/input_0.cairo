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
    data.append(FixedType { mag: 8192000, sign: false });
    data.append(FixedType { mag: 8192000, sign: true });
    data.append(FixedType { mag: 1900544, sign: false });
    data.append(FixedType { mag: 3538944, sign: true });
    data.append(FixedType { mag: 2228224, sign: true });
    data.append(FixedType { mag: 6356992, sign: false });
    data.append(FixedType { mag: 7143424, sign: false });
    data.append(FixedType { mag: 4587520, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}