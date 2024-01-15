use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 75497472, sign: false });
    data.append(FP8x23 { mag: 83886080, sign: false });
    data.append(FP8x23 { mag: 92274688, sign: false });
    data.append(FP8x23 { mag: 100663296, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
