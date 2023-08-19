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
    data.append(FixedType { mag: 8388608, sign: false });
    data.append(FixedType { mag: 16777216, sign: false });
    data.append(FixedType { mag: 25165824, sign: false });
    data.append(FixedType { mag: 33554432, sign: false });
    data.append(FixedType { mag: 41943040, sign: false });
    data.append(FixedType { mag: 50331648, sign: false });
    data.append(FixedType { mag: 58720256, sign: false });
    data.append(FixedType { mag: 67108864, sign: false });
    data.append(FixedType { mag: 75497472, sign: false });
    data.append(FixedType { mag: 83886080, sign: false });
    data.append(FixedType { mag: 92274688, sign: false });
    data.append(FixedType { mag: 100663296, sign: false });
    data.append(FixedType { mag: 109051904, sign: false });
    data.append(FixedType { mag: 117440512, sign: false });
    data.append(FixedType { mag: 125829120, sign: false });
    data.append(FixedType { mag: 134217728, sign: false });
    data.append(FixedType { mag: 142606336, sign: false });
    data.append(FixedType { mag: 150994944, sign: false });
    data.append(FixedType { mag: 159383552, sign: false });
    data.append(FixedType { mag: 167772160, sign: false });
    data.append(FixedType { mag: 176160768, sign: false });
    data.append(FixedType { mag: 184549376, sign: false });
    data.append(FixedType { mag: 192937984, sign: false });
    data.append(FixedType { mag: 201326592, sign: false });
    data.append(FixedType { mag: 209715200, sign: false });
    data.append(FixedType { mag: 218103808, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}