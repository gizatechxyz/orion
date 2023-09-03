use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
use orion::numbers::FP16x16;


fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 393216, sign: false });
    data.append(FP16x16 { mag: 458752, sign: false });
    data.append(FP16x16 { mag: 524288, sign: false });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 720896, sign: false });
    data.append(FP16x16 { mag: 786432, sign: false });
    data.append(FP16x16 { mag: 851968, sign: false });
    data.append(FP16x16 { mag: 917504, sign: false });
    data.append(FP16x16 { mag: 983040, sign: false });
    data.append(FP16x16 { mag: 1048576, sign: false });
    data.append(FP16x16 { mag: 1114112, sign: false });
    data.append(FP16x16 { mag: 1179648, sign: false });
    data.append(FP16x16 { mag: 1245184, sign: false });
    data.append(FP16x16 { mag: 1310720, sign: false });
    data.append(FP16x16 { mag: 1376256, sign: false });
    data.append(FP16x16 { mag: 1441792, sign: false });
    data.append(FP16x16 { mag: 1507328, sign: false });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 1638400, sign: false });
    data.append(FP16x16 { mag: 1703936, sign: false });
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

    
    TensorTrait::new(shape.span(), data.span())
}