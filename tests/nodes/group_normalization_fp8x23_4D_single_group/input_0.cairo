use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 10462599, sign: true });
    data.append(FP8x23 { mag: 6423082, sign: true });
    data.append(FP8x23 { mag: 3214070, sign: true });
    data.append(FP8x23 { mag: 15162586, sign: false });
    data.append(FP8x23 { mag: 5411424, sign: true });
    data.append(FP8x23 { mag: 18755948, sign: false });
    data.append(FP8x23 { mag: 9437388, sign: false });
    data.append(FP8x23 { mag: 3646371, sign: true });
    data.append(FP8x23 { mag: 6022929, sign: true });
    data.append(FP8x23 { mag: 13097515, sign: false });
    data.append(FP8x23 { mag: 15143336, sign: false });
    data.append(FP8x23 { mag: 4262506, sign: false });
    data.append(FP8x23 { mag: 4385511, sign: false });
    data.append(FP8x23 { mag: 6669492, sign: true });
    data.append(FP8x23 { mag: 12209711, sign: false });
    data.append(FP8x23 { mag: 8079416, sign: false });
    data.append(FP8x23 { mag: 8100121, sign: true });
    data.append(FP8x23 { mag: 1350764, sign: true });
    data.append(FP8x23 { mag: 2361456, sign: false });
    data.append(FP8x23 { mag: 11299532, sign: true });
    data.append(FP8x23 { mag: 4747131, sign: true });
    data.append(FP8x23 { mag: 7627056, sign: false });
    data.append(FP8x23 { mag: 5797891, sign: true });
    data.append(FP8x23 { mag: 9884868, sign: false });
    data.append(FP8x23 { mag: 568848, sign: true });
    data.append(FP8x23 { mag: 1792906, sign: true });
    data.append(FP8x23 { mag: 3159292, sign: true });
    data.append(FP8x23 { mag: 2306762, sign: true });
    data.append(FP8x23 { mag: 6188450, sign: true });
    data.append(FP8x23 { mag: 9593584, sign: false });
    data.append(FP8x23 { mag: 12958535, sign: true });
    data.append(FP8x23 { mag: 12924332, sign: false });
    data.append(FP8x23 { mag: 256345, sign: false });
    data.append(FP8x23 { mag: 10896865, sign: false });
    data.append(FP8x23 { mag: 1873112, sign: true });
    data.append(FP8x23 { mag: 1854541, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
