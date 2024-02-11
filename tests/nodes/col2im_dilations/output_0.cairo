use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(6);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 524288, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 1048576, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 1179648, sign: false });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 1703936, sign: false });
    data.append(FP16x16 { mag: 2097152, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 2228224, sign: false });
    data.append(FP16x16 { mag: 1245184, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 1310720, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
