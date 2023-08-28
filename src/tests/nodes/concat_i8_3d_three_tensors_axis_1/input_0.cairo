use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn input_0() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedType { mag: 0, sign: false });
    data.append(FixedType { mag: 1, sign: false });
    data.append(FixedType { mag: 2, sign: false });
    data.append(FixedType { mag: 3, sign: false });
    data.append(FixedType { mag: 4, sign: false });
    data.append(FixedType { mag: 5, sign: false });
    data.append(FixedType { mag: 6, sign: false });
    data.append(FixedType { mag: 7, sign: false });
    data.append(FixedType { mag: 8, sign: false });
    data.append(FixedType { mag: 9, sign: false });
    data.append(FixedType { mag: 10, sign: false });
    data.append(FixedType { mag: 11, sign: false });
    data.append(FixedType { mag: 12, sign: false });
    data.append(FixedType { mag: 13, sign: false });
    data.append(FixedType { mag: 14, sign: false });
    data.append(FixedType { mag: 15, sign: false });
    data.append(FixedType { mag: 16, sign: false });
    data.append(FixedType { mag: 17, sign: false });
    data.append(FixedType { mag: 18, sign: false });
    data.append(FixedType { mag: 19, sign: false });
    data.append(FixedType { mag: 20, sign: false });
    data.append(FixedType { mag: 21, sign: false });
    data.append(FixedType { mag: 22, sign: false });
    data.append(FixedType { mag: 23, sign: false });
    data.append(FixedType { mag: 24, sign: false });
    data.append(FixedType { mag: 25, sign: false });
    data.append(FixedType { mag: 26, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}