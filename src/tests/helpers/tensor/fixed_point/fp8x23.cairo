use array::ArrayTrait;
use array::SpanTrait;
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::impl_8x23;
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

// 1D
fn fp_tensor_1x3_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_1x3_neg_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

// 2D

fn fp_tensor_2x2_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_2x2_neg_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_3x3_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(5, false));
    data.append(FixedTrait::new_unscaled(6, false));
    data.append(FixedTrait::new_unscaled(7, false));
    data.append(FixedTrait::new_unscaled(8, false));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_3x3_neg_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    data.append(FixedTrait::new_unscaled(4, true));
    data.append(FixedTrait::new_unscaled(5, true));
    data.append(FixedTrait::new_unscaled(6, true));
    data.append(FixedTrait::new_unscaled(7, true));
    data.append(FixedTrait::new_unscaled(8, true));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_3x2_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(5, false));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_3x2_neg_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    data.append(FixedTrait::new_unscaled(4, true));
    data.append(FixedTrait::new_unscaled(5, true));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_3x1_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_3x1_neg_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(1);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_2x3_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(5, false));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_2x3_neg_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    data.append(FixedTrait::new_unscaled(4, true));
    data.append(FixedTrait::new_unscaled(5, true));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

// 3D

fn fp_tensor_2x2x2_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(5, false));
    data.append(FixedTrait::new_unscaled(6, false));
    data.append(FixedTrait::new_unscaled(7, false));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_2x2x2_neg_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    data.append(FixedTrait::new_unscaled(4, true));
    data.append(FixedTrait::new_unscaled(5, true));
    data.append(FixedTrait::new_unscaled(6, true));
    data.append(FixedTrait::new_unscaled(7, true));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_3x2x2_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(5, false));
    data.append(FixedTrait::new_unscaled(6, false));
    data.append(FixedTrait::new_unscaled(7, false));
    data.append(FixedTrait::new_unscaled(8, false));
    data.append(FixedTrait::new_unscaled(9, false));
    data.append(FixedTrait::new_unscaled(10, false));
    data.append(FixedTrait::new_unscaled(11, false));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

fn fp_tensor_3x2x2_neg_helper() -> Tensor<FixedType> {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(3, true));
    data.append(FixedTrait::new_unscaled(4, true));
    data.append(FixedTrait::new_unscaled(5, true));
    data.append(FixedTrait::new_unscaled(6, true));
    data.append(FixedTrait::new_unscaled(7, true));
    data.append(FixedTrait::new_unscaled(8, true));
    data.append(FixedTrait::new_unscaled(9, true));
    data.append(FixedTrait::new_unscaled(10, true));
    data.append(FixedTrait::new_unscaled(11, true));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), Option::Some(extra));

    return tensor;
}

