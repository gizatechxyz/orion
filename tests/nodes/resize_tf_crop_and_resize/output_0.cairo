use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 498073, sign: false });
    data.append(FP16x16 { mag: 517734, sign: false });
    data.append(FP16x16 { mag: 537395, sign: false });
    data.append(FP16x16 { mag: 576716, sign: false });
    data.append(FP16x16 { mag: 596377, sign: false });
    data.append(FP16x16 { mag: 616038, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 675020, sign: false });
    data.append(FP16x16 { mag: 694681, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
