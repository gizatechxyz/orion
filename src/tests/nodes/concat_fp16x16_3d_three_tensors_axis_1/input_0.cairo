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
    data.append(FixedType { mag: 65536, sign: false });
    data.append(FixedType { mag: 131072, sign: false });
    data.append(FixedType { mag: 196608, sign: false });
    data.append(FixedType { mag: 262144, sign: false });
    data.append(FixedType { mag: 327680, sign: false });
    data.append(FixedType { mag: 393216, sign: false });
    data.append(FixedType { mag: 458752, sign: false });
    data.append(FixedType { mag: 524288, sign: false });
    data.append(FixedType { mag: 589824, sign: false });
    data.append(FixedType { mag: 655360, sign: false });
    data.append(FixedType { mag: 720896, sign: false });
    data.append(FixedType { mag: 786432, sign: false });
    data.append(FixedType { mag: 851968, sign: false });
    data.append(FixedType { mag: 917504, sign: false });
    data.append(FixedType { mag: 983040, sign: false });
    data.append(FixedType { mag: 1048576, sign: false });
    data.append(FixedType { mag: 1114112, sign: false });
    data.append(FixedType { mag: 1179648, sign: false });
    data.append(FixedType { mag: 1245184, sign: false });
    data.append(FixedType { mag: 1310720, sign: false });
    data.append(FixedType { mag: 1376256, sign: false });
    data.append(FixedType { mag: 1441792, sign: false });
    data.append(FixedType { mag: 1507328, sign: false });
    data.append(FixedType { mag: 1572864, sign: false });
    data.append(FixedType { mag: 1638400, sign: false });
    data.append(FixedType { mag: 1703936, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}