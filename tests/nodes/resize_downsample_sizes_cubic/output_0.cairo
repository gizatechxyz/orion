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
    data.append(FP16x16 { mag: 106875, sign: false });
    data.append(FP16x16 { mag: 196911, sign: false });
    data.append(FP16x16 { mag: 286947, sign: false });
    data.append(FP16x16 { mag: 467019, sign: false });
    data.append(FP16x16 { mag: 557056, sign: false });
    data.append(FP16x16 { mag: 647092, sign: false });
    data.append(FP16x16 { mag: 827164, sign: false });
    data.append(FP16x16 { mag: 917200, sign: false });
    data.append(FP16x16 { mag: 1007236, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
