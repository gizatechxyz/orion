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
    data.append(FixedType { mag: 7143424, sign: false });
    data.append(FixedType { mag: 7471104, sign: true });
    data.append(FixedType { mag: 262144, sign: false });
    data.append(FixedType { mag: 1835008, sign: false });
    data.append(FixedType { mag: 2883584, sign: false });
    data.append(FixedType { mag: 3014656, sign: true });
    data.append(FixedType { mag: 3145728, sign: true });
    data.append(FixedType { mag: 4521984, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}