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
    data.append(FP16x16 { mag: 1400832, sign: true });
    data.append(FP16x16 { mag: 85184, sign: true });
    data.append(FP16x16 { mag: 399616, sign: false });
    data.append(FP16x16 { mag: 1828864, sign: false });
    data.append(FP16x16 { mag: 962560, sign: true });
    data.append(FP16x16 { mag: 1190912, sign: false });
    data.append(FP16x16 { mag: 1355776, sign: true });
    data.append(FP16x16 { mag: 946176, sign: false });
    data.append(FP16x16 { mag: 502528, sign: true });
    data.append(FP16x16 { mag: 85824, sign: false });
    data.append(FP16x16 { mag: 1333248, sign: false });
    data.append(FP16x16 { mag: 39200, sign: true });
    data.append(FP16x16 { mag: 712192, sign: false });
    data.append(FP16x16 { mag: 1497088, sign: false });
    data.append(FP16x16 { mag: 1521664, sign: false });
    data.append(FP16x16 { mag: 606720, sign: true });
    data.append(FP16x16 { mag: 848384, sign: false });
    data.append(FP16x16 { mag: 1732608, sign: true });
    data.append(FP16x16 { mag: 1158144, sign: true });
    data.append(FP16x16 { mag: 1806336, sign: false });
    data.append(FP16x16 { mag: 935424, sign: true });
    data.append(FP16x16 { mag: 1106944, sign: true });
    data.append(FP16x16 { mag: 1180672, sign: true });
    data.append(FP16x16 { mag: 1509376, sign: true });
    data.append(FP16x16 { mag: 856064, sign: false });
    data.append(FP16x16 { mag: 1841152, sign: false });
    data.append(FP16x16 { mag: 75008, sign: false });
    data.append(FP16x16 { mag: 19504, sign: true });
    data.append(FP16x16 { mag: 1326080, sign: false });
    data.append(FP16x16 { mag: 1423360, sign: true });
    data.append(FP16x16 { mag: 1258496, sign: true });
    data.append(FP16x16 { mag: 671232, sign: true });
    data.append(FP16x16 { mag: 548352, sign: false });
    data.append(FP16x16 { mag: 797696, sign: true });
    data.append(FP16x16 { mag: 1245184, sign: true });
    data.append(FP16x16 { mag: 1404928, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
