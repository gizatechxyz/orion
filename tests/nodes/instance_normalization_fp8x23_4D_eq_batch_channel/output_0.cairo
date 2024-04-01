use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 4546062, sign: true });
    data.append(FP8x23 { mag: 4173193, sign: true });
    data.append(FP8x23 { mag: 10422706, sign: true });
    data.append(FP8x23 { mag: 1822120, sign: true });
    data.append(FP8x23 { mag: 14251254, sign: true });
    data.append(FP8x23 { mag: 11773163, sign: true });
    data.append(FP8x23 { mag: 181332, sign: true });
    data.append(FP8x23 { mag: 327476, sign: true });
    data.append(FP8x23 { mag: 4201185, sign: true });
    data.append(FP8x23 { mag: 3103412, sign: false });
    data.append(FP8x23 { mag: 16516863, sign: true });
    data.append(FP8x23 { mag: 3910327, sign: true });
    data.append(FP8x23 { mag: 1581872, sign: true });
    data.append(FP8x23 { mag: 3481641, sign: true });
    data.append(FP8x23 { mag: 3137940, sign: true });
    data.append(FP8x23 { mag: 718853, sign: true });
    data.append(FP8x23 { mag: 7059, sign: true });
    data.append(FP8x23 { mag: 1077457, sign: true });
    data.append(FP8x23 { mag: 1470134, sign: true });
    data.append(FP8x23 { mag: 2526843, sign: true });
    data.append(FP8x23 { mag: 4004837, sign: true });
    data.append(FP8x23 { mag: 2292716, sign: true });
    data.append(FP8x23 { mag: 2316395, sign: true });
    data.append(FP8x23 { mag: 1777181, sign: true });
    data.append(FP8x23 { mag: 5943457, sign: true });
    data.append(FP8x23 { mag: 4717080, sign: true });
    data.append(FP8x23 { mag: 3458604, sign: true });
    data.append(FP8x23 { mag: 13422854, sign: true });
    data.append(FP8x23 { mag: 1791802, sign: false });
    data.append(FP8x23 { mag: 3734989, sign: true });
    data.append(FP8x23 { mag: 2103929, sign: true });
    data.append(FP8x23 { mag: 1181530, sign: true });
    data.append(FP8x23 { mag: 2340185, sign: true });
    data.append(FP8x23 { mag: 4816997, sign: true });
    data.append(FP8x23 { mag: 20422696, sign: true });
    data.append(FP8x23 { mag: 8671750, sign: true });
    data.append(FP8x23 { mag: 3106938, sign: true });
    data.append(FP8x23 { mag: 3719844, sign: true });
    data.append(FP8x23 { mag: 1715363, sign: true });
    data.append(FP8x23 { mag: 261986, sign: false });
    data.append(FP8x23 { mag: 3325528, sign: true });
    data.append(FP8x23 { mag: 1297987, sign: true });
    data.append(FP8x23 { mag: 2470474, sign: true });
    data.append(FP8x23 { mag: 1508976, sign: true });
    data.append(FP8x23 { mag: 500811, sign: true });
    data.append(FP8x23 { mag: 2459250, sign: true });
    data.append(FP8x23 { mag: 1987322, sign: true });
    data.append(FP8x23 { mag: 2562423, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
