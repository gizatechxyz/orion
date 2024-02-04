use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8254028, sign: false });
    data.append(FP8x23 { mag: 15690628, sign: false });
    data.append(FP8x23 { mag: 5515815, sign: false });
    data.append(FP8x23 { mag: 5436207, sign: false });
    data.append(FP8x23 { mag: 4444335, sign: false });
    data.append(FP8x23 { mag: 1122784, sign: false });
    data.append(FP8x23 { mag: 12881007, sign: false });
    data.append(FP8x23 { mag: 2781310, sign: true });
    data.append(FP8x23 { mag: 582197, sign: false });
    data.append(FP8x23 { mag: 2891656, sign: false });
    data.append(FP8x23 { mag: 1856695, sign: true });
    data.append(FP8x23 { mag: 2058841, sign: false });
    data.append(FP8x23 { mag: 1014957, sign: true });
    data.append(FP8x23 { mag: 9724722, sign: true });
    data.append(FP8x23 { mag: 22497248, sign: true });
    data.append(FP8x23 { mag: 12912647, sign: false });
    data.append(FP8x23 { mag: 10905704, sign: true });
    data.append(FP8x23 { mag: 3390802, sign: false });
    data.append(FP8x23 { mag: 14589767, sign: true });
    data.append(FP8x23 { mag: 8773874, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
