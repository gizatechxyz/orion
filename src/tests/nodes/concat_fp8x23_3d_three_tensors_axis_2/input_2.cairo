use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn input_2() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedType { mag: 452984832, sign: false });
    data.append(FixedType { mag: 461373440, sign: false });
    data.append(FixedType { mag: 469762048, sign: false });
    data.append(FixedType { mag: 478150656, sign: false });
    data.append(FixedType { mag: 486539264, sign: false });
    data.append(FixedType { mag: 494927872, sign: false });
    data.append(FixedType { mag: 503316480, sign: false });
    data.append(FixedType { mag: 511705088, sign: false });
    data.append(FixedType { mag: 520093696, sign: false });
    data.append(FixedType { mag: 528482304, sign: false });
    data.append(FixedType { mag: 536870912, sign: false });
    data.append(FixedType { mag: 545259520, sign: false });
    data.append(FixedType { mag: 553648128, sign: false });
    data.append(FixedType { mag: 562036736, sign: false });
    data.append(FixedType { mag: 570425344, sign: false });
    data.append(FixedType { mag: 578813952, sign: false });
    data.append(FixedType { mag: 587202560, sign: false });
    data.append(FixedType { mag: 595591168, sign: false });
    data.append(FixedType { mag: 603979776, sign: false });
    data.append(FixedType { mag: 612368384, sign: false });
    data.append(FixedType { mag: 620756992, sign: false });
    data.append(FixedType { mag: 629145600, sign: false });
    data.append(FixedType { mag: 637534208, sign: false });
    data.append(FixedType { mag: 645922816, sign: false });
    data.append(FixedType { mag: 654311424, sign: false });
    data.append(FixedType { mag: 662700032, sign: false });
    data.append(FixedType { mag: 671088640, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}