use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 41143, sign: true });
    data.append(FP16x16 { mag: 51803, sign: false });
    data.append(FP16x16 { mag: 113556, sign: false });
    data.append(FP16x16 { mag: 64774, sign: false });
    data.append(FP16x16 { mag: 866, sign: false });
    data.append(FP16x16 { mag: 698, sign: true });
    data.append(FP16x16 { mag: 106500, sign: false });
    data.append(FP16x16 { mag: 98929, sign: false });
    data.append(FP16x16 { mag: 7551, sign: false });
    data.append(FP16x16 { mag: 30689, sign: true });
    data.append(FP16x16 { mag: 38325, sign: false });
    data.append(FP16x16 { mag: 48164, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
