use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn input_1() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedType { mag: 1236715, sign: false });
    data.append(FixedType { mag: 4771319, sign: false });
    data.append(FixedType { mag: 8392691, sign: false });
    data.append(FixedType { mag: 36629024, sign: true });
    data.append(FixedType { mag: 34768195, sign: false });
    data.append(FixedType { mag: 2858178, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}