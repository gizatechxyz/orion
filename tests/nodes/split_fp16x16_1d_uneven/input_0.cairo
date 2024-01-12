use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(7);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5701632, sign: true });
    data.append(FP16x16 { mag: 6946816, sign: false });
    data.append(FP16x16 { mag: 2883584, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 2883584, sign: true });
    data.append(FP16x16 { mag: 6160384, sign: false });
    data.append(FP16x16 { mag: 8257536, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
