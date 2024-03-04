use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(7);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2655, sign: false });
    data.append(FP16x16 { mag: 27005, sign: false });
    data.append(FP16x16 { mag: 28670, sign: false });
    data.append(FP16x16 { mag: 49392, sign: false });
    data.append(FP16x16 { mag: 39855, sign: false });
    data.append(FP16x16 { mag: 37109, sign: false });
    data.append(FP16x16 { mag: 58515, sign: false });
    data.append(FP16x16 { mag: 40588, sign: false });
    data.append(FP16x16 { mag: 59543, sign: false });
    data.append(FP16x16 { mag: 58423, sign: false });
    data.append(FP16x16 { mag: 200, sign: false });
    data.append(FP16x16 { mag: 57208, sign: false });
    data.append(FP16x16 { mag: 10967, sign: false });
    data.append(FP16x16 { mag: 21516, sign: false });
    data.append(FP16x16 { mag: 4197, sign: false });
    data.append(FP16x16 { mag: 2419, sign: false });
    data.append(FP16x16 { mag: 15655, sign: false });
    data.append(FP16x16 { mag: 33193, sign: false });
    data.append(FP16x16 { mag: 40116, sign: false });
    data.append(FP16x16 { mag: 14725, sign: false });
    data.append(FP16x16 { mag: 37526, sign: false });
    data.append(FP16x16 { mag: 4098, sign: false });
    data.append(FP16x16 { mag: 45267, sign: false });
    data.append(FP16x16 { mag: 11802, sign: false });
    data.append(FP16x16 { mag: 53114, sign: false });
    data.append(FP16x16 { mag: 53602, sign: false });
    data.append(FP16x16 { mag: 23812, sign: false });
    data.append(FP16x16 { mag: 50714, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
