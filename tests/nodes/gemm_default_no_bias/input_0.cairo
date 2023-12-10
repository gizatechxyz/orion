use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 48671, sign: false });
    data.append(FP16x16 { mag: 53291, sign: false });
    data.append(FP16x16 { mag: 61962, sign: false });
    data.append(FP16x16 { mag: 23548, sign: false });
    data.append(FP16x16 { mag: 12042, sign: false });
    data.append(FP16x16 { mag: 198, sign: false });
    data.append(FP16x16 { mag: 26605, sign: false });
    data.append(FP16x16 { mag: 42749, sign: false });
    data.append(FP16x16 { mag: 42426, sign: false });
    data.append(FP16x16 { mag: 16917, sign: false });
    data.append(FP16x16 { mag: 50488, sign: false });
    data.append(FP16x16 { mag: 10785, sign: false });
    data.append(FP16x16 { mag: 63703, sign: false });
    data.append(FP16x16 { mag: 16964, sign: false });
    data.append(FP16x16 { mag: 24102, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
