use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 989855744, sign: true });
    data.append(FP8x23 { mag: 100663296, sign: false });
    data.append(FP8x23 { mag: 956301312, sign: false });
    data.append(FP8x23 { mag: 285212672, sign: false });
    data.append(FP8x23 { mag: 662700032, sign: false });
    data.append(FP8x23 { mag: 469762048, sign: true });
    data.append(FP8x23 { mag: 360710144, sign: true });
    data.append(FP8x23 { mag: 729808896, sign: true });
    data.append(FP8x23 { mag: 998244352, sign: false });
    data.append(FP8x23 { mag: 905969664, sign: true });
    data.append(FP8x23 { mag: 201326592, sign: false });
    data.append(FP8x23 { mag: 419430400, sign: true });
    data.append(FP8x23 { mag: 704643072, sign: false });
    data.append(FP8x23 { mag: 327155712, sign: false });
    data.append(FP8x23 { mag: 704643072, sign: false });
    data.append(FP8x23 { mag: 905969664, sign: false });
    data.append(FP8x23 { mag: 704643072, sign: false });
    data.append(FP8x23 { mag: 738197504, sign: true });
    data.append(FP8x23 { mag: 343932928, sign: true });
    data.append(FP8x23 { mag: 620756992, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
