use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 89661, sign: false });
    data.append(FP16x16 { mag: 174948, sign: false });
    data.append(FP16x16 { mag: 263018, sign: false });
    data.append(FP16x16 { mag: 430809, sign: false });
    data.append(FP16x16 { mag: 516096, sign: false });
    data.append(FP16x16 { mag: 604165, sign: false });
    data.append(FP16x16 { mag: 783087, sign: false });
    data.append(FP16x16 { mag: 868374, sign: false });
    data.append(FP16x16 { mag: 956443, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
