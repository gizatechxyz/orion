use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 62000, sign: false });
    data.append(FP16x16 { mag: 64560, sign: false });
    data.append(FP16x16 { mag: 23724, sign: false });
    data.append(FP16x16 { mag: 53487, sign: false });
    data.append(FP16x16 { mag: 44710, sign: false });
    data.append(FP16x16 { mag: 54528, sign: false });
    data.append(FP16x16 { mag: 39071, sign: false });
    data.append(FP16x16 { mag: 3222, sign: false });
    data.append(FP16x16 { mag: 22465, sign: false });
    data.append(FP16x16 { mag: 56410, sign: false });
    data.append(FP16x16 { mag: 11578, sign: false });
    data.append(FP16x16 { mag: 57495, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
