use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 60818, sign: false });
    data.append(FP16x16 { mag: 59979, sign: false });
    data.append(FP16x16 { mag: 36912, sign: false });
    data.append(FP16x16 { mag: 46433, sign: false });
    data.append(FP16x16 { mag: 3745, sign: false });
    data.append(FP16x16 { mag: 29174, sign: false });
    data.append(FP16x16 { mag: 3555, sign: false });
    data.append(FP16x16 { mag: 3092, sign: false });
    data.append(FP16x16 { mag: 60956, sign: false });
    data.append(FP16x16 { mag: 37111, sign: false });
    data.append(FP16x16 { mag: 28077, sign: false });
    data.append(FP16x16 { mag: 9904, sign: false });
    data.append(FP16x16 { mag: 17752, sign: false });
    data.append(FP16x16 { mag: 51564, sign: false });
    data.append(FP16x16 { mag: 16512, sign: false });
    data.append(FP16x16 { mag: 18193, sign: false });
    data.append(FP16x16 { mag: 62859, sign: false });
    data.append(FP16x16 { mag: 28772, sign: false });
    data.append(FP16x16 { mag: 42434, sign: false });
    data.append(FP16x16 { mag: 12591, sign: false });
    data.append(FP16x16 { mag: 24303, sign: false });
    data.append(FP16x16 { mag: 19725, sign: false });
    data.append(FP16x16 { mag: 14636, sign: false });
    data.append(FP16x16 { mag: 57618, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
