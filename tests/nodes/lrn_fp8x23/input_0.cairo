use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 157361506, sign: false });
    data.append(FP8x23 { mag: 65947556, sign: false });
    data.append(FP8x23 { mag: 184958702, sign: true });
    data.append(FP8x23 { mag: 121472923, sign: false });
    data.append(FP8x23 { mag: 180907185, sign: true });
    data.append(FP8x23 { mag: 2217758, sign: false });
    data.append(FP8x23 { mag: 103839573, sign: true });
    data.append(FP8x23 { mag: 220429352, sign: false });
    data.append(FP8x23 { mag: 229977910, sign: true });
    data.append(FP8x23 { mag: 190142360, sign: false });
    data.append(FP8x23 { mag: 80417469, sign: false });
    data.append(FP8x23 { mag: 233216920, sign: true });
    data.append(FP8x23 { mag: 23249529, sign: false });
    data.append(FP8x23 { mag: 55705405, sign: false });
    data.append(FP8x23 { mag: 249837752, sign: false });
    data.append(FP8x23 { mag: 184155206, sign: true });
    data.append(FP8x23 { mag: 71269904, sign: true });
    data.append(FP8x23 { mag: 109102145, sign: true });
    data.append(FP8x23 { mag: 164635440, sign: true });
    data.append(FP8x23 { mag: 50362409, sign: true });
    data.append(FP8x23 { mag: 213977248, sign: true });
    data.append(FP8x23 { mag: 176365894, sign: true });
    data.append(FP8x23 { mag: 119986317, sign: false });
    data.append(FP8x23 { mag: 57149935, sign: true });
    data.append(FP8x23 { mag: 83110014, sign: true });
    data.append(FP8x23 { mag: 116234081, sign: false });
    data.append(FP8x23 { mag: 173484779, sign: true });
    data.append(FP8x23 { mag: 176794109, sign: true });
    data.append(FP8x23 { mag: 16253546, sign: false });
    data.append(FP8x23 { mag: 147203787, sign: false });
    data.append(FP8x23 { mag: 144635843, sign: true });
    data.append(FP8x23 { mag: 188210025, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
