use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 9596129, sign: false });
    data.append(FP8x23 { mag: 8524695, sign: false });
    data.append(FP8x23 { mag: 8030491, sign: true });
    data.append(FP8x23 { mag: 8640310, sign: false });
    data.append(FP8x23 { mag: 12854812, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
