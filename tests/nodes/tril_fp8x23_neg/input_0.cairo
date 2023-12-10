use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 520093696, sign: true });
    data.append(FP8x23 { mag: 444596224, sign: false });
    data.append(FP8x23 { mag: 293601280, sign: true });
    data.append(FP8x23 { mag: 218103808, sign: true });
    data.append(FP8x23 { mag: 595591168, sign: true });
    data.append(FP8x23 { mag: 989855744, sign: false });
    data.append(FP8x23 { mag: 411041792, sign: true });
    data.append(FP8x23 { mag: 1023410176, sign: false });
    data.append(FP8x23 { mag: 276824064, sign: true });
    data.append(FP8x23 { mag: 704643072, sign: true });
    data.append(FP8x23 { mag: 452984832, sign: true });
    data.append(FP8x23 { mag: 813694976, sign: true });
    data.append(FP8x23 { mag: 822083584, sign: false });
    data.append(FP8x23 { mag: 578813952, sign: false });
    data.append(FP8x23 { mag: 427819008, sign: true });
    data.append(FP8x23 { mag: 92274688, sign: true });
    data.append(FP8x23 { mag: 788529152, sign: false });
    data.append(FP8x23 { mag: 981467136, sign: false });
    data.append(FP8x23 { mag: 201326592, sign: true });
    data.append(FP8x23 { mag: 813694976, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
