use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 21210813, sign: true });
    data.append(FP8x23 { mag: 18026313, sign: true });
    data.append(FP8x23 { mag: 11180685, sign: false });
    data.append(FP8x23 { mag: 9192264, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
