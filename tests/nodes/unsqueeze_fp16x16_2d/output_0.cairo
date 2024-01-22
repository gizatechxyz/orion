use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(2);
    shape.append(4);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5046272, sign: false });
    data.append(FP16x16 { mag: 6946816, sign: false });
    data.append(FP16x16 { mag: 4063232, sign: false });
    data.append(FP16x16 { mag: 4259840, sign: false });
    data.append(FP16x16 { mag: 2555904, sign: true });
    data.append(FP16x16 { mag: 1900544, sign: false });
    data.append(FP16x16 { mag: 3604480, sign: true });
    data.append(FP16x16 { mag: 6881280, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
