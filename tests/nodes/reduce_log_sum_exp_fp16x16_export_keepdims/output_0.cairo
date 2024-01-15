use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16W> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 393216, sign: false });
    data.append(FP16x16 { mag: 524288, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 786432, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
