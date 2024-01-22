use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5046272, sign: true });
    data.append(FP16x16 { mag: 1245184, sign: false });
    data.append(FP16x16 { mag: 1835008, sign: true });
    data.append(FP16x16 { mag: 1703936, sign: true });
    data.append(FP16x16 { mag: 2621440, sign: true });
    data.append(FP16x16 { mag: 5701632, sign: false });
    data.append(FP16x16 { mag: 4063232, sign: true });
    data.append(FP16x16 { mag: 1835008, sign: true });
    data.append(FP16x16 { mag: 3604480, sign: true });
    data.append(FP16x16 { mag: 4259840, sign: false });
    data.append(FP16x16 { mag: 4063232, sign: false });
    data.append(FP16x16 { mag: 2293760, sign: false });
    data.append(FP16x16 { mag: 393216, sign: true });
    data.append(FP16x16 { mag: 4784128, sign: false });
    data.append(FP16x16 { mag: 1179648, sign: false });
    data.append(FP16x16 { mag: 2162688, sign: true });
    data.append(FP16x16 { mag: 2555904, sign: false });
    data.append(FP16x16 { mag: 1835008, sign: true });
    data.append(FP16x16 { mag: 3145728, sign: false });
    data.append(FP16x16 { mag: 2293760, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
