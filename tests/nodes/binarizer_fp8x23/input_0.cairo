use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 19910639, sign: true });
    data.append(FP8x23 { mag: 16200109, sign: false });
    data.append(FP8x23 { mag: 7318221, sign: false });
    data.append(FP8x23 { mag: 3839710, sign: true });
    data.append(FP8x23 { mag: 5198439, sign: true });
    data.append(FP8x23 { mag: 13440550, sign: false });
    data.append(FP8x23 { mag: 3475736, sign: false });
    data.append(FP8x23 { mag: 12659448, sign: false });
    data.append(FP8x23 { mag: 22040361, sign: true });
    data.append(FP8x23 { mag: 16960032, sign: true });
    data.append(FP8x23 { mag: 20980250, sign: false });
    data.append(FP8x23 { mag: 19528339, sign: true });
    data.append(FP8x23 { mag: 24567413, sign: true });
    data.append(FP8x23 { mag: 3374986, sign: true });
    data.append(FP8x23 { mag: 5428617, sign: false });
    data.append(FP8x23 { mag: 16499249, sign: true });
    data.append(FP8x23 { mag: 14111693, sign: false });
    data.append(FP8x23 { mag: 22806062, sign: false });
    data.append(FP8x23 { mag: 5020958, sign: false });
    data.append(FP8x23 { mag: 20749322, sign: false });
    data.append(FP8x23 { mag: 7503709, sign: true });
    data.append(FP8x23 { mag: 24435895, sign: false });
    data.append(FP8x23 { mag: 12933406, sign: true });
    data.append(FP8x23 { mag: 11401085, sign: true });
    data.append(FP8x23 { mag: 97331, sign: false });
    data.append(FP8x23 { mag: 21216642, sign: true });
    data.append(FP8x23 { mag: 2018862, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
