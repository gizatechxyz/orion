use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 24824, sign: false });
    data.append(FP16x16 { mag: 17409, sign: false });
    data.append(FP16x16 { mag: 32722, sign: false });
    data.append(FP16x16 { mag: 28536, sign: false });
    data.append(FP16x16 { mag: 35635, sign: false });
    data.append(FP16x16 { mag: 25964, sign: false });
    data.append(FP16x16 { mag: 31473, sign: false });
    data.append(FP16x16 { mag: 43310, sign: false });
    data.append(FP16x16 { mag: 33179, sign: false });
    data.append(FP16x16 { mag: 49819, sign: false });
    data.append(FP16x16 { mag: 19988, sign: false });
    data.append(FP16x16 { mag: 14397, sign: false });
    data.append(FP16x16 { mag: 27673, sign: false });
    data.append(FP16x16 { mag: 21189, sign: false });
    data.append(FP16x16 { mag: 30484, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
