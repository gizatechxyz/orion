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
    data.append(FixedType { mag: 1769472, sign: false });
    data.append(FixedType { mag: 1835008, sign: false });
    data.append(FixedType { mag: 1900544, sign: false });
    data.append(FixedType { mag: 1966080, sign: false });
    data.append(FixedType { mag: 2031616, sign: false });
    data.append(FixedType { mag: 2097152, sign: false });
    data.append(FixedType { mag: 2162688, sign: false });
    data.append(FixedType { mag: 2228224, sign: false });
    data.append(FixedType { mag: 2293760, sign: false });
    data.append(FixedType { mag: 2359296, sign: false });
    data.append(FixedType { mag: 2424832, sign: false });
    data.append(FixedType { mag: 2490368, sign: false });
    data.append(FixedType { mag: 2555904, sign: false });
    data.append(FixedType { mag: 2621440, sign: false });
    data.append(FixedType { mag: 2686976, sign: false });
    data.append(FixedType { mag: 2752512, sign: false });
    data.append(FixedType { mag: 2818048, sign: false });
    data.append(FixedType { mag: 2883584, sign: false });
    data.append(FixedType { mag: 2949120, sign: false });
    data.append(FixedType { mag: 3014656, sign: false });
    data.append(FixedType { mag: 3080192, sign: false });
    data.append(FixedType { mag: 3145728, sign: false });
    data.append(FixedType { mag: 3211264, sign: false });
    data.append(FixedType { mag: 3276800, sign: false });
    data.append(FixedType { mag: 3342336, sign: false });
    data.append(FixedType { mag: 3407872, sign: false });
    data.append(FixedType { mag: 3473408, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}