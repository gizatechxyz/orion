use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(18);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 2424832, sign: true });
    data.append(FP16x16 { mag: 851968, sign: true });
    data.append(FP16x16 { mag: 3538944, sign: true });
    data.append(FP16x16 { mag: 7864320, sign: false });
    data.append(FP16x16 { mag: 131072, sign: true });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(FP16x16 { mag: 6750208, sign: false });
    data.append(FP16x16 { mag: 7405568, sign: false });
    data.append(FP16x16 { mag: 1245184, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 7929856, sign: true });
    data.append(FP16x16 { mag: 6488064, sign: false });
    data.append(FP16x16 { mag: 3407872, sign: true });
    data.append(FP16x16 { mag: 2293760, sign: true });
    data.append(FP16x16 { mag: 1966080, sign: false });
    data.append(FP16x16 { mag: 2031616, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
