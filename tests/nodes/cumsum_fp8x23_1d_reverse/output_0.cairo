use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 125829120, sign: false });
    data.append(FP8x23 { mag: 117440512, sign: false });
    data.append(FP8x23 { mag: 100663296, sign: false });
    data.append(FP8x23 { mag: 75497472, sign: false });
    data.append(FP8x23 { mag: 41943040, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
