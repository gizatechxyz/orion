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
    data.append(FixedType { mag: 461373440, sign: false });
    data.append(FixedType { mag: 67108864, sign: false });
    data.append(FixedType { mag: 394264576, sign: true });
    data.append(FixedType { mag: 494927872, sign: true });
    data.append(FixedType { mag: 503316480, sign: true });
    data.append(FixedType { mag: 570425344, sign: false });
    data.append(FixedType { mag: 469762048, sign: false });
    data.append(FixedType { mag: 276824064, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}