use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 53724, sign: false });
    data.append(FP16x16 { mag: 100114, sign: true });
    data.append(FP16x16 { mag: 105707, sign: true });
    data.append(FP16x16 { mag: 172713, sign: false });
    data.append(FP16x16 { mag: 117489, sign: true });
    data.append(FP16x16 { mag: 131055, sign: false });
    data.append(FP16x16 { mag: 136968, sign: true });
    data.append(FP16x16 { mag: 139039, sign: true });
    data.append(FP16x16 { mag: 179765, sign: false });
    data.append(FP16x16 { mag: 116549, sign: false });
    data.append(FP16x16 { mag: 86012, sign: true });
    data.append(FP16x16 { mag: 42842, sign: false });
    data.append(FP16x16 { mag: 102196, sign: false });
    data.append(FP16x16 { mag: 91319, sign: true });
    data.append(FP16x16 { mag: 154563, sign: true });
    data.append(FP16x16 { mag: 148265, sign: false });
    data.append(FP16x16 { mag: 147485, sign: false });
    data.append(FP16x16 { mag: 18844, sign: true });
    data.append(FP16x16 { mag: 87916, sign: true });
    data.append(FP16x16 { mag: 53116, sign: false });
    data.append(FP16x16 { mag: 189184, sign: false });
    data.append(FP16x16 { mag: 172959, sign: true });
    data.append(FP16x16 { mag: 24790, sign: false });
    data.append(FP16x16 { mag: 141694, sign: false });
    data.append(FP16x16 { mag: 142845, sign: false });
    data.append(FP16x16 { mag: 88179, sign: false });
    data.append(FP16x16 { mag: 76572, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
