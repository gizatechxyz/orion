use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 15671515, sign: true });
    data.append(FP8x23 { mag: 18137844, sign: true });
    data.append(FP8x23 { mag: 16645152, sign: true });
    data.append(FP8x23 { mag: 17341110, sign: true });
    data.append(FP8x23 { mag: 1755482, sign: false });
    data.append(FP8x23 { mag: 30694908, sign: true });
    data.append(FP8x23 { mag: 12371686, sign: true });
    data.append(FP8x23 { mag: 6981271, sign: true });
    data.append(FP8x23 { mag: 4726310, sign: false });
    data.append(FP8x23 { mag: 9101548, sign: true });
    data.append(FP8x23 { mag: 20213250, sign: true });
    data.append(FP8x23 { mag: 18436168, sign: false });
    data.append(FP8x23 { mag: 18898560, sign: true });
    data.append(FP8x23 { mag: 15432130, sign: true });
    data.append(FP8x23 { mag: 18789194, sign: true });
    data.append(FP8x23 { mag: 16147842, sign: true });
    data.append(FP8x23 { mag: 30279632, sign: true });
    data.append(FP8x23 { mag: 1921934, sign: true });
    data.append(FP8x23 { mag: 15320543, sign: true });
    data.append(FP8x23 { mag: 3240866, sign: true });
    data.append(FP8x23 { mag: 1396860, sign: true });
    data.append(FP8x23 { mag: 4199834, sign: true });
    data.append(FP8x23 { mag: 10403702, sign: true });
    data.append(FP8x23 { mag: 20791298, sign: false });
    data.append(FP8x23 { mag: 18010106, sign: true });
    data.append(FP8x23 { mag: 19320158, sign: true });
    data.append(FP8x23 { mag: 16717802, sign: true });
    data.append(FP8x23 { mag: 16419005, sign: true });
    data.append(FP8x23 { mag: 4506460, sign: true });
    data.append(FP8x23 { mag: 33066696, sign: true });
    data.append(FP8x23 { mag: 1202829, sign: false });
    data.append(FP8x23 { mag: 15310373, sign: true });
    data.append(FP8x23 { mag: 8237352, sign: true });
    data.append(FP8x23 { mag: 12886606, sign: false });
    data.append(FP8x23 { mag: 15345274, sign: false });
    data.append(FP8x23 { mag: 4912907, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
