use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 125498, sign: false });
    data.append(FP16x16 { mag: 54402, sign: false });
    data.append(FP16x16 { mag: 70165, sign: false });
    data.append(FP16x16 { mag: 64638, sign: false });
    data.append(FP16x16 { mag: 77787, sign: false });
    data.append(FP16x16 { mag: 3710, sign: true });
    data.append(FP16x16 { mag: 41451, sign: false });
    data.append(FP16x16 { mag: 30523, sign: false });
    data.append(FP16x16 { mag: 46300, sign: false });
    data.append(FP16x16 { mag: 50725, sign: false });
    data.append(FP16x16 { mag: 1879, sign: false });
    data.append(FP16x16 { mag: 31403, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
