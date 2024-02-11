use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 786432, sign: false });
    data.append(FP16x16 { mag: 1769472, sign: false });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 4128768, sign: false });
    data.append(FP16x16 { mag: 7077888, sign: false });
    data.append(FP16x16 { mag: 5308416, sign: false });
    data.append(FP16x16 { mag: 4718592, sign: false });
    data.append(FP16x16 { mag: 7667712, sign: false });
    data.append(FP16x16 { mag: 5505024, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
