use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 109153, sign: false });
    data.append(FP16x16 { mag: 177293, sign: false });
    data.append(FP16x16 { mag: 156959, sign: false });
    data.append(FP16x16 { mag: 132883, sign: false });
    data.append(FP16x16 { mag: 73584, sign: false });
    data.append(FP16x16 { mag: 6199, sign: false });
    data.append(FP16x16 { mag: 18979, sign: true });
    data.append(FP16x16 { mag: 45601, sign: false });
    data.append(FP16x16 { mag: 53176, sign: false });
    data.append(FP16x16 { mag: 10705, sign: false });
    data.append(FP16x16 { mag: 41333, sign: false });
    data.append(FP16x16 { mag: 43153, sign: false });
    data.append(FP16x16 { mag: 91938, sign: false });
    data.append(FP16x16 { mag: 163454, sign: false });
    data.append(FP16x16 { mag: 79246, sign: false });
    data.append(FP16x16 { mag: 159821, sign: false });
    data.append(FP16x16 { mag: 33449, sign: false });
    data.append(FP16x16 { mag: 1753, sign: false });
    data.append(FP16x16 { mag: 29942, sign: true });
    data.append(FP16x16 { mag: 8953, sign: false });
    data.append(FP16x16 { mag: 95624, sign: false });
    data.append(FP16x16 { mag: 6060, sign: true });
    data.append(FP16x16 { mag: 63720, sign: false });
    data.append(FP16x16 { mag: 82768, sign: false });
    data.append(FP16x16 { mag: 158933, sign: false });
    data.append(FP16x16 { mag: 167317, sign: false });
    data.append(FP16x16 { mag: 117437, sign: false });
    data.append(FP16x16 { mag: 214698, sign: false });
    data.append(FP16x16 { mag: 15136, sign: false });
    data.append(FP16x16 { mag: 23110, sign: false });
    data.append(FP16x16 { mag: 19480, sign: false });
    data.append(FP16x16 { mag: 15119, sign: false });
    data.append(FP16x16 { mag: 64567, sign: false });
    data.append(FP16x16 { mag: 58490, sign: false });
    data.append(FP16x16 { mag: 88960, sign: false });
    data.append(FP16x16 { mag: 186594, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
