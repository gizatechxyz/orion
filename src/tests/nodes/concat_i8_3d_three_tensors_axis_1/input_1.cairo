use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn input_1() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedType { mag: 27, sign: false });
    data.append(FixedType { mag: 28, sign: false });
    data.append(FixedType { mag: 29, sign: false });
    data.append(FixedType { mag: 30, sign: false });
    data.append(FixedType { mag: 31, sign: false });
    data.append(FixedType { mag: 32, sign: false });
    data.append(FixedType { mag: 33, sign: false });
    data.append(FixedType { mag: 34, sign: false });
    data.append(FixedType { mag: 35, sign: false });
    data.append(FixedType { mag: 36, sign: false });
    data.append(FixedType { mag: 37, sign: false });
    data.append(FixedType { mag: 38, sign: false });
    data.append(FixedType { mag: 39, sign: false });
    data.append(FixedType { mag: 40, sign: false });
    data.append(FixedType { mag: 41, sign: false });
    data.append(FixedType { mag: 42, sign: false });
    data.append(FixedType { mag: 43, sign: false });
    data.append(FixedType { mag: 44, sign: false });
    data.append(FixedType { mag: 45, sign: false });
    data.append(FixedType { mag: 46, sign: false });
    data.append(FixedType { mag: 47, sign: false });
    data.append(FixedType { mag: 48, sign: false });
    data.append(FixedType { mag: 49, sign: false });
    data.append(FixedType { mag: 50, sign: false });
    data.append(FixedType { mag: 51, sign: false });
    data.append(FixedType { mag: 52, sign: false });
    data.append(FixedType { mag: 53, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}