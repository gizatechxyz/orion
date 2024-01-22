use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 39653, sign: false });
    data.append(FP16x16 { mag: 1049, sign: false });
    data.append(FP16x16 { mag: 50921, sign: false });
    data.append(FP16x16 { mag: 49433, sign: false });
    data.append(FP16x16 { mag: 42267, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
