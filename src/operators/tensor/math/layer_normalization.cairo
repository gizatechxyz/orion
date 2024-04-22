use core::option::OptionTrait;
use core::traits::TryInto;
use orion::numbers::{NumberTrait, I32IntoU32};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};
use orion::operators::tensor::{
    TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor, BoolTensor
};
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};

/// Cf: TensorTrait::layer_normalization docstring
fn layer_normalization<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +Div<Tensor<T>>,
    +Sub<Tensor<T>>,
    +Add<Tensor<T>>,
    +Mul<Tensor<T>>,
    +Into<usize, MAG>,
>(
    self: @Tensor<T>,
    scale: @Tensor<T>,
    B: Option<@Tensor<T>>,
    axis: Option<i32>,
    epsilon: Option<T>,
    stash_type: Option<usize>,
) -> (Tensor<T>, Tensor<T>, Tensor<T>) {
    let X_rank = (*self).shape.len();

    let mut axis = match axis {
        Option::Some(axis) => axis,
        Option::None => -1,
    };
    let epsilon = match epsilon {
        Option::Some(epsilon) => epsilon,
        Option::None => NumberTrait::zero(), // default of onnx is 1e-05 
    };

    let axis = if axis < 0 {
        X_rank - axis.into()
    } else {
        axis.into()
    };

    let unsqueezed_rank = X_rank - axis;
    let mut reduction_shape = array![];
    let mut i = 0;
    while i != axis {
        reduction_shape.append(*(*self).shape.at(i));
        i += 1;
    };

    let mut i = 0;
    while i != unsqueezed_rank {
        reduction_shape.append(1);
        i += 1;
    };

    let mut row_number = 1;
    let mut col_number = 1;
    let mut i = 0;
    while i != X_rank {
        if i < axis {
            row_number *= *(*self).shape.at(i);
        } else {
            col_number *= *(*self).shape.at(i);
        }

        i += 1;
    };

    let mut shape_matrix = array![];
    shape_matrix.append(row_number.try_into().unwrap());
    shape_matrix.append(col_number.try_into().unwrap());

    // Shape [1, 1] to mutiply one element tensors with 2D matrices
    let mut shape_one = array![];
    shape_one.append(1);
    shape_one.append(1);

    let mut col_number_tensor = array![];
    col_number_tensor.append(NumberTrait::new_unscaled(col_number.into(), false));

    let mut epsilon_tensor = array![];
    epsilon_tensor.append(epsilon);

    let mut one_tensor = array![];
    one_tensor.append(NumberTrait::one());

    let x_mat = self.reshape(shape_matrix.span(), false);
    let x_mean = x_mat
        .reduce_sum(Option::Some(array![1].span()), Option::Some(true), Option::Some(false))
        / TensorTrait::new(shape_one.span(), col_number_tensor.span());

    let x_diff = x_mat - x_mean;
    let x_squared_diff = x_diff * x_diff;

    let variance = x_squared_diff
        .reduce_sum(Option::Some(array![1].span()), Option::Some(true), Option::Some(false))
        / TensorTrait::new(shape_one.span(), col_number_tensor.span());
    let variance_eps = variance + TensorTrait::new(shape_one.span(), epsilon_tensor.span());

    let std_dev = variance_eps.sqrt();

    let inv_std_dev = TensorTrait::new(shape_one.span(), one_tensor.span()) / std_dev;

    let y_mat = x_diff * inv_std_dev;

    let scale = if (*scale).shape.len() < (*self).shape.len() {
        // Append 1 in scale shape to make sure scale has a dimension compatible with Y for multiplication
        let mut shape = array![];
        let mut i = 0;
        while i != (*self).shape.len() - (*scale).shape.len() {
            shape.append(1);
            i += 1;
        };

        let mut i = 0;
        while i != (*scale).shape.len() {
            shape.append(*(*scale).shape.at(i));
            i += 1;
        };

        TensorTrait::new(shape.span(), (*scale).data)
    } else {
        *scale
    };

    let mut i = 0;
    let mut target_shape: Array<i32> = array![];
    while i < (*self)
        .shape
        .len() {
            target_shape.append((*(*self).shape.at(i)).try_into().unwrap());
            i += 1;
        };
    let Y = y_mat.reshape(target_shape.span(), false) * scale;

    let Y = match B {
        Option::Some(B) => {
            let B = if (*B).shape.len() < (*self).shape.len() {
                // Append 1 in B shape to make sure scale has a dimension compatible with Y for multiplication
                let mut shape = array![];
                let mut i = 0;
                while i != (*self).shape.len() - (*B).shape.len() {
                    shape.append(1);
                    i += 1;
                };

                let mut i = 0;
                while i != (*B).shape.len() {
                    shape.append(*(*B).shape.at(i));
                    i += 1;
                };

                TensorTrait::new(shape.span(), (*B).data)
            } else {
                *B
            };
            Y + B
        },
        Option::None => Y,
    };

    let X_mean = TensorTrait::new(reduction_shape.span(), x_mean.data);
    let X_inv_std_dev = TensorTrait::new(reduction_shape.span(), inv_std_dev.data);

    (Y, X_mean, X_inv_std_dev)
}

