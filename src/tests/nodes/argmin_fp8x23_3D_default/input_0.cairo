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
    data.append(FixedType { mag: 1015021568, sign: true });
    data.append(FixedType { mag: 394264576, sign: false });
    data.append(FixedType { mag: 377487360, sign: true });
    data.append(FixedType { mag: 276824064, sign: true });
    data.append(FixedType { mag: 545259520, sign: true });
    data.append(FixedType { mag: 243269632, sign: false });
    data.append(FixedType { mag: 771751936, sign: false });
    data.append(FixedType { mag: 243269632, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}