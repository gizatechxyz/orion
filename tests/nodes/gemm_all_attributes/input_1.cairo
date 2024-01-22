use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 23725, sign: false });
    data.append(FP16x16 { mag: 17645, sign: false });
    data.append(FP16x16 { mag: 43346, sign: false });
    data.append(FP16x16 { mag: 23010, sign: false });
    data.append(FP16x16 { mag: 58499, sign: false });
    data.append(FP16x16 { mag: 53494, sign: false });
    data.append(FP16x16 { mag: 8855, sign: false });
    data.append(FP16x16 { mag: 52549, sign: false });
    data.append(FP16x16 { mag: 33829, sign: false });
    data.append(FP16x16 { mag: 64693, sign: false });
    data.append(FP16x16 { mag: 23894, sign: false });
    data.append(FP16x16 { mag: 27926, sign: false });
    data.append(FP16x16 { mag: 3015, sign: false });
    data.append(FP16x16 { mag: 63591, sign: false });
    data.append(FP16x16 { mag: 31838, sign: false });
    data.append(FP16x16 { mag: 6987, sign: false });
    data.append(FP16x16 { mag: 63224, sign: false });
    data.append(FP16x16 { mag: 63842, sign: false });
    data.append(FP16x16 { mag: 27830, sign: false });
    data.append(FP16x16 { mag: 56137, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
