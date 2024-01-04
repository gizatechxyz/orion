use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 729808896, sign: false });
    data.append(FP8x23 { mag: 864026624, sign: false });
    data.append(FP8x23 { mag: 360710144, sign: true });
    data.append(FP8x23 { mag: 360710144, sign: false });
    data.append(FP8x23 { mag: 109051904, sign: true });
    data.append(FP8x23 { mag: 545259520, sign: false });
    data.append(FP8x23 { mag: 33554432, sign: true });
    data.append(FP8x23 { mag: 436207616, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
