use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 10596986, sign: false });
    data.append(FP8x23 { mag: 13797276, sign: true });
    data.append(FP8x23 { mag: 726032, sign: true });
    data.append(FP8x23 { mag: 2944650, sign: true });
    data.append(FP8x23 { mag: 5288885, sign: true });
    data.append(FP8x23 { mag: 12046768, sign: true });
    data.append(FP8x23 { mag: 3375686, sign: false });
    data.append(FP8x23 { mag: 8744354, sign: true });
    data.append(FP8x23 { mag: 8940485, sign: false });
    data.append(FP8x23 { mag: 6541405, sign: true });
    data.append(FP8x23 { mag: 3256492, sign: false });
    data.append(FP8x23 { mag: 6889087, sign: false });
    data.append(FP8x23 { mag: 2560312, sign: true });
    data.append(FP8x23 { mag: 9717397, sign: true });
    data.append(FP8x23 { mag: 8774793, sign: true });
    data.append(FP8x23 { mag: 893052, sign: true });
    data.append(FP8x23 { mag: 7995400, sign: false });
    data.append(FP8x23 { mag: 9505615, sign: true });
    data.append(FP8x23 { mag: 541572, sign: false });
    data.append(FP8x23 { mag: 13005167, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
