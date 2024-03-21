use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3683900, sign: true });
    data.append(FP8x23 { mag: 12214077, sign: false });
    data.append(FP8x23 { mag: 8413897, sign: true });
    data.append(FP8x23 { mag: 13466313, sign: true });
    data.append(FP8x23 { mag: 568041, sign: false });
    data.append(FP8x23 { mag: 6012992, sign: false });
    data.append(FP8x23 { mag: 481851, sign: true });
    data.append(FP8x23 { mag: 2780836, sign: false });
    data.append(FP8x23 { mag: 8216124, sign: false });
    data.append(FP8x23 { mag: 7374084, sign: true });
    data.append(FP8x23 { mag: 1717446, sign: true });
    data.append(FP8x23 { mag: 4711634, sign: false });
    data.append(FP8x23 { mag: 4638506, sign: false });
    data.append(FP8x23 { mag: 8648521, sign: false });
    data.append(FP8x23 { mag: 3876580, sign: false });
    data.append(FP8x23 { mag: 4334610, sign: false });
    data.append(FP8x23 { mag: 4047245, sign: false });
    data.append(FP8x23 { mag: 8391439, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
