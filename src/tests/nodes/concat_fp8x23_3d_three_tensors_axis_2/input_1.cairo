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
    data.append(FixedType { mag: 226492416, sign: false });
    data.append(FixedType { mag: 234881024, sign: false });
    data.append(FixedType { mag: 243269632, sign: false });
    data.append(FixedType { mag: 251658240, sign: false });
    data.append(FixedType { mag: 260046848, sign: false });
    data.append(FixedType { mag: 268435456, sign: false });
    data.append(FixedType { mag: 276824064, sign: false });
    data.append(FixedType { mag: 285212672, sign: false });
    data.append(FixedType { mag: 293601280, sign: false });
    data.append(FixedType { mag: 301989888, sign: false });
    data.append(FixedType { mag: 310378496, sign: false });
    data.append(FixedType { mag: 318767104, sign: false });
    data.append(FixedType { mag: 327155712, sign: false });
    data.append(FixedType { mag: 335544320, sign: false });
    data.append(FixedType { mag: 343932928, sign: false });
    data.append(FixedType { mag: 352321536, sign: false });
    data.append(FixedType { mag: 360710144, sign: false });
    data.append(FixedType { mag: 369098752, sign: false });
    data.append(FixedType { mag: 377487360, sign: false });
    data.append(FixedType { mag: 385875968, sign: false });
    data.append(FixedType { mag: 394264576, sign: false });
    data.append(FixedType { mag: 402653184, sign: false });
    data.append(FixedType { mag: 411041792, sign: false });
    data.append(FixedType { mag: 419430400, sign: false });
    data.append(FixedType { mag: 427819008, sign: false });
    data.append(FixedType { mag: 436207616, sign: false });
    data.append(FixedType { mag: 444596224, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}