use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn input_0() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedType { mag: 16576692, sign: false });
    data.append(FixedType { mag: 41797636, sign: false });
    data.append(FixedType { mag: 10487655, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}