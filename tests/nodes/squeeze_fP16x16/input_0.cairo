use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 10092544, sign: false });
    data.append(FP16x16 { mag: 14024704, sign: false });
    data.append(FP16x16 { mag: 14090240, sign: false });
    data.append(FP16x16 { mag: 5373952, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
