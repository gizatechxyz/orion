use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 184064, sign: true });
    data.append(FP16x16 { mag: 130304, sign: false });
    data.append(FP16x16 { mag: 117248, sign: true });
    data.append(FP16x16 { mag: 94016, sign: false });
    data.append(FP16x16 { mag: 147072, sign: false });
    data.append(FP16x16 { mag: 100800, sign: true });
    data.append(FP16x16 { mag: 84544, sign: false });
    data.append(FP16x16 { mag: 74496, sign: false });
    data.append(FP16x16 { mag: 71616, sign: false });
    data.append(FP16x16 { mag: 69760, sign: false });
    data.append(FP16x16 { mag: 112832, sign: false });
    data.append(FP16x16 { mag: 52928, sign: false });
    data.append(FP16x16 { mag: 175488, sign: true });
    data.append(FP16x16 { mag: 6936, sign: true });
    data.append(FP16x16 { mag: 193664, sign: true });
    data.append(FP16x16 { mag: 39648, sign: false });
    data.append(FP16x16 { mag: 166528, sign: true });
    data.append(FP16x16 { mag: 180096, sign: false });
    data.append(FP16x16 { mag: 130944, sign: true });
    data.append(FP16x16 { mag: 105792, sign: false });
    data.append(FP16x16 { mag: 51392, sign: true });
    data.append(FP16x16 { mag: 45408, sign: true });
    data.append(FP16x16 { mag: 169344, sign: true });
    data.append(FP16x16 { mag: 151936, sign: true });
    data.append(FP16x16 { mag: 90112, sign: true });
    data.append(FP16x16 { mag: 9808, sign: true });
    data.append(FP16x16 { mag: 98368, sign: true });
    data.append(FP16x16 { mag: 179584, sign: true });
    data.append(FP16x16 { mag: 122048, sign: false });
    data.append(FP16x16 { mag: 19856, sign: false });
    data.append(FP16x16 { mag: 38944, sign: true });
    data.append(FP16x16 { mag: 65792, sign: true });
    data.append(FP16x16 { mag: 187136, sign: false });
    data.append(FP16x16 { mag: 190336, sign: false });
    data.append(FP16x16 { mag: 119744, sign: true });
    data.append(FP16x16 { mag: 128832, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
