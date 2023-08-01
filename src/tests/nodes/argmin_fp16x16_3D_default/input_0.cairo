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
    data.append(FixedType { mag: 7995392, sign: true });
    data.append(FixedType { mag: 5439488, sign: true });
    data.append(FixedType { mag: 1310720, sign: true });
    data.append(FixedType { mag: 5963776, sign: false });
    data.append(FixedType { mag: 5570560, sign: false });
    data.append(FixedType { mag: 4521984, sign: true });
    data.append(FixedType { mag: 2359296, sign: false });
    data.append(FixedType { mag: 2097152, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}