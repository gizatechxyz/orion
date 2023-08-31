use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::FixedImpl;
use orion::operators::tensor::implementations::tensor_i32_fp16x16::Tensor_i32_fp16x16;
use orion::numbers::{i32, FP16x16};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 4, sign: true });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 6, sign: true });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 4, sign: true });
    data.append(i32 { mag: 0, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 4, sign: true });
    data.append(i32 { mag: 4, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 4, sign: true });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 0, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}