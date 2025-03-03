use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8540492, sign: false });
    data.append(FP8x23 { mag: 9382786, sign: true });
    data.append(FP8x23 { mag: 6586859, sign: false });
    data.append(FP8x23 { mag: 17804984, sign: false });
    data.append(FP8x23 { mag: 1660556, sign: false });
    data.append(FP8x23 { mag: 2773353, sign: false });
    data.append(FP8x23 { mag: 3068045, sign: false });
    data.append(FP8x23 { mag: 976243, sign: false });
    data.append(FP8x23 { mag: 573526, sign: false });
    data.append(FP8x23 { mag: 7273832, sign: false });
    data.append(FP8x23 { mag: 1552350, sign: true });
    data.append(FP8x23 { mag: 7584258, sign: true });
    data.append(FP8x23 { mag: 5499422, sign: true });
    data.append(FP8x23 { mag: 11491652, sign: true });
    data.append(FP8x23 { mag: 3165269, sign: false });
    data.append(FP8x23 { mag: 5968577, sign: true });
    data.append(FP8x23 { mag: 3539512, sign: false });
    data.append(FP8x23 { mag: 8542260, sign: false });
    data.append(FP8x23 { mag: 5040973, sign: false });
    data.append(FP8x23 { mag: 4504122, sign: true });
    data.append(FP8x23 { mag: 22596482, sign: false });
    data.append(FP8x23 { mag: 12034543, sign: true });
    data.append(FP8x23 { mag: 11339130, sign: true });
    data.append(FP8x23 { mag: 3209119, sign: true });
    data.append(FP8x23 { mag: 11559511, sign: false });
    data.append(FP8x23 { mag: 214059, sign: false });
    data.append(FP8x23 { mag: 24245974, sign: true });
    data.append(FP8x23 { mag: 5981880, sign: true });
    data.append(FP8x23 { mag: 7071770, sign: true });
    data.append(FP8x23 { mag: 7802461, sign: true });
    data.append(FP8x23 { mag: 2514734, sign: false });
    data.append(FP8x23 { mag: 3869380, sign: true });
    data.append(FP8x23 { mag: 1170766, sign: false });
    data.append(FP8x23 { mag: 11821055, sign: false });
    data.append(FP8x23 { mag: 4197997, sign: true });
    data.append(FP8x23 { mag: 15530508, sign: false });
    data.append(FP8x23 { mag: 10691257, sign: false });
    data.append(FP8x23 { mag: 60018, sign: false });
    data.append(FP8x23 { mag: 5375578, sign: false });
    data.append(FP8x23 { mag: 2139229, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
