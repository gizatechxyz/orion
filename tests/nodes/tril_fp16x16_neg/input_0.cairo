use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 720896, sign: true });
    data.append(FP16x16 { mag: 5242880, sign: false });
    data.append(FP16x16 { mag: 2424832, sign: true });
    data.append(FP16x16 { mag: 5701632, sign: true });
    data.append(FP16x16 { mag: 4456448, sign: false });
    data.append(FP16x16 { mag: 7405568, sign: true });
    data.append(FP16x16 { mag: 917504, sign: true });
    data.append(FP16x16 { mag: 131072, sign: true });
    data.append(FP16x16 { mag: 1048576, sign: true });
    data.append(FP16x16 { mag: 8060928, sign: true });
    data.append(FP16x16 { mag: 2359296, sign: true });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 5898240, sign: false });
    data.append(FP16x16 { mag: 1966080, sign: true });
    data.append(FP16x16 { mag: 720896, sign: false });
    data.append(FP16x16 { mag: 5177344, sign: true });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 3670016, sign: true });
    data.append(FP16x16 { mag: 7602176, sign: true });
    data.append(FP16x16 { mag: 7012352, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
