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
    data.append(FixedType { mag: 528482304, sign: true });
    data.append(FixedType { mag: 796917760, sign: false });
    data.append(FixedType { mag: 301989888, sign: true });
    data.append(FixedType { mag: 989855744, sign: false });
    data.append(FixedType { mag: 385875968, sign: false });
    data.append(FixedType { mag: 629145600, sign: true });
    data.append(FixedType { mag: 830472192, sign: false });
    data.append(FixedType { mag: 436207616, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}