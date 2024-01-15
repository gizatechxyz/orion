use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 9020, sign: false });
    data.append(FP16x16 { mag: 1508, sign: false });
    data.append(FP16x16 { mag: 17596, sign: false });
    data.append(FP16x16 { mag: 15725, sign: false });
    data.append(FP16x16 { mag: 17557, sign: false });
    data.append(FP16x16 { mag: 6286, sign: false });
    data.append(FP16x16 { mag: 33993, sign: false });
    data.append(FP16x16 { mag: 24966, sign: false });
    data.append(FP16x16 { mag: 18648, sign: false });
    data.append(FP16x16 { mag: 1482, sign: false });
    data.append(FP16x16 { mag: 56402, sign: false });
    data.append(FP16x16 { mag: 11205, sign: false });
    data.append(FP16x16 { mag: 59749, sign: false });
    data.append(FP16x16 { mag: 32628, sign: false });
    data.append(FP16x16 { mag: 51145, sign: false });
    data.append(FP16x16 { mag: 37477, sign: false });
    data.append(FP16x16 { mag: 14287, sign: false });
    data.append(FP16x16 { mag: 18084, sign: false });
    data.append(FP16x16 { mag: 13969, sign: false });
    data.append(FP16x16 { mag: 52688, sign: false });
    data.append(FP16x16 { mag: 40536, sign: false });
    data.append(FP16x16 { mag: 36430, sign: false });
    data.append(FP16x16 { mag: 17009, sign: false });
    data.append(FP16x16 { mag: 50657, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
