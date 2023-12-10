use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 196608, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
