use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7274496, sign: true });
    data.append(FP16x16 { mag: 1835008, sign: false });
    data.append(FP16x16 { mag: 131072, sign: true });
    data.append(FP16x16 { mag: 4194304, sign: true });
    data.append(FP16x16 { mag: 393216, sign: false });
    data.append(FP16x16 { mag: 2097152, sign: true });
    data.append(FP16x16 { mag: 917504, sign: true });
    data.append(FP16x16 { mag: 7208960, sign: false });
    data.append(FP16x16 { mag: 8323072, sign: true });
    data.append(FP16x16 { mag: 458752, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 3538944, sign: true });
    data.append(FP16x16 { mag: 7929856, sign: false });
    data.append(FP16x16 { mag: 393216, sign: true });
    data.append(FP16x16 { mag: 5767168, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 3211264, sign: true });
    data.append(FP16x16 { mag: 6488064, sign: false });
    data.append(FP16x16 { mag: 851968, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
