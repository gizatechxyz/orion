use core::traits::TryInto;
use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;
use core::traits::Into;
use orion::numbers::NumberTrait;
use orion::operators::tensor::{
    TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor, BoolTensor
};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};
use core::debug::PrintTrait;
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};


fn group_normalization<T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +Div<T>,
    +Sub<T>,
    +Neg<T>,
    +Div<Tensor<T>>,
    +Sub<Tensor<T>>,
    +Add<Tensor<T>>,
    +Mul<Tensor<T>>,
    +Into<usize, MAG>,>
    (
    self: @Tensor<T>,
    num_groups: usize,
    scale: @Tensor<T>,
    bias: @Tensor<T>,
    epsilon: Option<T>,) -> Tensor<T> {


    let tensor_rank = (*self).shape.len();
    let zero = NumberTrait::zero() ;  

    let mut epsilon = match epsilon {
        Option::Some(epsilon) => epsilon,
        Option::None => zero
    };

    assert(*(*self.shape).at(1) % num_groups == 0, 'numgroup indivisible by channel');

    let group_size = *(*self.shape).at(1) / num_groups;

    let mut new_shape: Array<usize> = array![];
    new_shape.append((*(*self.shape).at(0)));
    new_shape.append(num_groups);
    new_shape.append(group_size);

    let mut i: usize = 2;
    loop {
        if (i >= tensor_rank) {
            break;
        }
        new_shape.append((*(*self.shape).at(i)));
        i += 1;
    };
    let x_reshaped = self.reshape(target_shape: new_shape.span());

    let mut axes: Array<usize> = array![];
    let mut i: usize = 2;
    loop {
        if (i >= new_shape.len()) {
            break;
        }
        axes.append(i);
        i += 1;
    };

    let mut mean = x_reshaped.reduce_mean( axes: Option::Some(axes.span()),
    keepdims: Option::Some((true)),
    noop_with_empty_axes: Option::None(()));
    let x_diff = x_reshaped - mean;
    let x_diff_squared = x_diff * x_diff;
    let mut variance = x_diff_squared.reduce_mean(axes: Option::Some(axes.span()),keepdims: Option::Some((true)),noop_with_empty_axes: Option::None(()));

    // adjust shape of epsilon tensor to match the shape of variance tensor
    let mut epsilon = TensorTrait::new(shape: array![].span(), data: array![epsilon].span());
    let mut epsilon_shape: Array<usize> = array![];
    let mut i: usize = 0;
    loop {
        if (i >= variance.shape.len()) {
            break;
        }
        epsilon_shape.append(1);
        i += 1;
    };
    epsilon = epsilon.reshape(target_shape: epsilon_shape.span());
    let mut std = (variance + epsilon).sqrt();

    let mut bool_tensor = false;
    let mut i: usize = 0;
    loop {
        if (i >= std.data.len()) {
            break;
        }
        if *std.data.at(i) == zero {
            bool_tensor = true;
        }
        i += 1;
    };

    if bool_tensor == true {
        // clip values to min_std_val to avoid possible division by zero errors
        let mut a: usize = 500;
        let mut min_std_val = NumberTrait::<T, MAG>::half() / NumberTrait::<T, MAG>::new_unscaled(a.into(), false); 
        std = std.clip(min: Option::Some((min_std_val)), max: Option::None(()), );
    };

    let mut x_normalized = x_diff / std;
    x_normalized = x_normalized.reshape(target_shape: *self.shape);

    let mut dim_ones: Array<usize> = array![];
    let mut i: usize = 0;
    loop {
        if i >= (*self.shape).len() - 2 {
            break;
        }
        dim_ones.append(1);
        i += 1;
    };

    let mut dim_ones_clone = dim_ones.clone();
    let mut new_scale_shape: Array<usize> = array![];
    new_scale_shape.append(*(*scale.shape).at(0));
    let mut i: usize = 0;
    loop {
        if i >= dim_ones.len() {
            break;
        }
        new_scale_shape.append(*dim_ones.at(i));

        i += 1;
    };
 
    // bias and scale should have same shape
    let mut scale = scale.reshape(target_shape: new_scale_shape.span());
    let new_bias_shape = new_scale_shape.clone(); 
    let mut bias = bias.reshape(target_shape: new_bias_shape.span()); // since bias and scale have same shape

    // expanding the the dims accordingly to complete arthmetic ops for tensors of different shapes
    let mut expanded_new_scale_shape: Array<usize> = array![];
    if new_scale_shape.len() != x_normalized.shape.len() {
        let shape_diff = x_normalized.shape.len() - new_scale_shape.len();
        let mut i: usize = 0;
        loop {
            if i >= shape_diff {
                break;
            }
            expanded_new_scale_shape.append(1);
            i += 1;
        };
        expanded_new_scale_shape.append(*scale.shape.at(0));

        let mut i: usize = 0;
        loop {
            if i >= dim_ones_clone.len() {
                break;
            }
            expanded_new_scale_shape.append(*dim_ones_clone.at(i));

            i += 1;
        };
        
        let expanded_new_bias_shape = expanded_new_scale_shape.clone(); 

        scale = scale.reshape(target_shape: expanded_new_scale_shape.span());
        bias = bias.reshape(target_shape: expanded_new_bias_shape.span());
    }

    return x_normalized * scale + bias;
}
