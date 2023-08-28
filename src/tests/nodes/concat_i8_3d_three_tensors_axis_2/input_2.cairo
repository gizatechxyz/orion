use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn input_2() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedType { mag: 54, sign: false });
    data.append(FixedType { mag: 55, sign: false });
    data.append(FixedType { mag: 56, sign: false });
    data.append(FixedType { mag: 57, sign: false });
    data.append(FixedType { mag: 58, sign: false });
    data.append(FixedType { mag: 59, sign: false });
    data.append(FixedType { mag: 60, sign: false });
    data.append(FixedType { mag: 61, sign: false });
    data.append(FixedType { mag: 62, sign: false });
    data.append(FixedType { mag: 63, sign: false });
    data.append(FixedType { mag: 64, sign: false });
    data.append(FixedType { mag: 65, sign: false });
    data.append(FixedType { mag: 66, sign: false });
    data.append(FixedType { mag: 67, sign: false });
    data.append(FixedType { mag: 68, sign: false });
    data.append(FixedType { mag: 69, sign: false });
    data.append(FixedType { mag: 70, sign: false });
    data.append(FixedType { mag: 71, sign: false });
    data.append(FixedType { mag: 72, sign: false });
    data.append(FixedType { mag: 73, sign: false });
    data.append(FixedType { mag: 74, sign: false });
    data.append(FixedType { mag: 75, sign: false });
    data.append(FixedType { mag: 76, sign: false });
    data.append(FixedType { mag: 77, sign: false });
    data.append(FixedType { mag: 78, sign: false });
    data.append(FixedType { mag: 79, sign: false });
    data.append(FixedType { mag: 80, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}