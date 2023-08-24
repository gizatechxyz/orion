use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn input_0() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FixedType { mag: 131072, sign: true });
    data.append(FixedType { mag: 6488064, sign: true });
    data.append(FixedType { mag: 6750208, sign: false });
    data.append(FixedType { mag: 393216, sign: true });
    data.append(FixedType { mag: 7602176, sign: false });
    data.append(FixedType { mag: 5439488, sign: true });
    data.append(FixedType { mag: 3014656, sign: false });
    data.append(FixedType { mag: 2818048, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}