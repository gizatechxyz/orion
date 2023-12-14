use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 28531311, sign: false });
    data.append(FP8x23 { mag: 29330831, sign: false });
    data.append(FP8x23 { mag: 30060735, sign: false });
    data.append(FP8x23 { mag: 30732182, sign: false });
    data.append(FP8x23 { mag: 31353845, sign: false });
    data.append(FP8x23 { mag: 31932599, sign: false });
    data.append(FP8x23 { mag: 32473987, sign: false });
    data.append(FP8x23 { mag: 32982543, sign: false });
    data.append(FP8x23 { mag: 33462023, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
