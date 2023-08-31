use array::ArrayTrait;

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
use orion::operators::tensor::implementations::tensor_fp16x16::Tensor_fp16x16;
use orion::numbers::FP16x16;


fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1769472, sign: false });
    data.append(FP16x16 { mag: 1835008, sign: false });
    data.append(FP16x16 { mag: 1900544, sign: false });
    data.append(FP16x16 { mag: 1966080, sign: false });
    data.append(FP16x16 { mag: 2031616, sign: false });
    data.append(FP16x16 { mag: 2097152, sign: false });
    data.append(FP16x16 { mag: 2162688, sign: false });
    data.append(FP16x16 { mag: 2228224, sign: false });
    data.append(FP16x16 { mag: 2293760, sign: false });
    data.append(FP16x16 { mag: 2359296, sign: false });
    data.append(FP16x16 { mag: 2424832, sign: false });
    data.append(FP16x16 { mag: 2490368, sign: false });
    data.append(FP16x16 { mag: 2555904, sign: false });
    data.append(FP16x16 { mag: 2621440, sign: false });
    data.append(FP16x16 { mag: 2686976, sign: false });
    data.append(FP16x16 { mag: 2752512, sign: false });
    data.append(FP16x16 { mag: 2818048, sign: false });
    data.append(FP16x16 { mag: 2883584, sign: false });
    data.append(FP16x16 { mag: 2949120, sign: false });
    data.append(FP16x16 { mag: 3014656, sign: false });
    data.append(FP16x16 { mag: 3080192, sign: false });
    data.append(FP16x16 { mag: 3145728, sign: false });
    data.append(FP16x16 { mag: 3211264, sign: false });
    data.append(FP16x16 { mag: 3276800, sign: false });
    data.append(FP16x16 { mag: 3342336, sign: false });
    data.append(FP16x16 { mag: 3407872, sign: false });
    data.append(FP16x16 { mag: 3473408, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}