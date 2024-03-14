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
    data.append(FP16x16 { mag: 1092608, sign: false });
    data.append(FP16x16 { mag: 634368, sign: false });
    data.append(FP16x16 { mag: 1613824, sign: true });
    data.append(FP16x16 { mag: 947200, sign: false });
    data.append(FP16x16 { mag: 819200, sign: true });
    data.append(FP16x16 { mag: 1251328, sign: true });
    data.append(FP16x16 { mag: 1854464, sign: false });
    data.append(FP16x16 { mag: 1760256, sign: true });
    data.append(FP16x16 { mag: 135168, sign: false });
    data.append(FP16x16 { mag: 88064, sign: false });
    data.append(FP16x16 { mag: 924672, sign: true });
    data.append(FP16x16 { mag: 1415168, sign: false });
    data.append(FP16x16 { mag: 862208, sign: false });
    data.append(FP16x16 { mag: 673280, sign: false });
    data.append(FP16x16 { mag: 857600, sign: true });
    data.append(FP16x16 { mag: 1040384, sign: false });
    data.append(FP16x16 { mag: 1608704, sign: false });
    data.append(FP16x16 { mag: 1477632, sign: false });
    data.append(FP16x16 { mag: 1648640, sign: true });
    data.append(FP16x16 { mag: 1811456, sign: true });
    data.append(FP16x16 { mag: 582656, sign: true });
    data.append(FP16x16 { mag: 1102848, sign: false });
    data.append(FP16x16 { mag: 734208, sign: true });
    data.append(FP16x16 { mag: 1461248, sign: false });
    data.append(FP16x16 { mag: 151680, sign: true });
    data.append(FP16x16 { mag: 1010688, sign: true });
    data.append(FP16x16 { mag: 1299456, sign: true });
    data.append(FP16x16 { mag: 681472, sign: false });
    data.append(FP16x16 { mag: 418048, sign: true });
    data.append(FP16x16 { mag: 941056, sign: true });
    data.append(FP16x16 { mag: 1766400, sign: true });
    data.append(FP16x16 { mag: 1608704, sign: false });
    data.append(FP16x16 { mag: 890368, sign: false });
    data.append(FP16x16 { mag: 236800, sign: true });
    data.append(FP16x16 { mag: 1204224, sign: false });
    data.append(FP16x16 { mag: 554496, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
