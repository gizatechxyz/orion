use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 18988, sign: false });
    data.append(FP16x16 { mag: 19077, sign: true });
    data.append(FP16x16 { mag: 155860, sign: false });
    data.append(FP16x16 { mag: 9080, sign: false });
    data.append(FP16x16 { mag: 15261, sign: false });
    data.append(FP16x16 { mag: 15350, sign: true });
    data.append(FP16x16 { mag: 8016, sign: false });
    data.append(FP16x16 { mag: 156924, sign: false });
    data.append(FP16x16 { mag: 23315, sign: true });
    data.append(FP16x16 { mag: 23226, sign: false });
    data.append(FP16x16 { mag: 24955, sign: false });
    data.append(FP16x16 { mag: 139985, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
