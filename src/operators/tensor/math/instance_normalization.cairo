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


fn instance_normalization<T,
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
    +PrintTrait<T>,
    +Div<Tensor<T>>,
    +Sub<Tensor<T>>,
    +Add<Tensor<T>>,
    +Mul<Tensor<T>>,
    +Into<usize, MAG>,>
    (
    self: @Tensor<T>,
    scale: @Tensor<T>,
    bias: @Tensor<T>,
    epsilon: Option<T>,) -> Tensor<T> {


    let dim_x = (*self).shape.len();
    let zero = NumberTrait::zero() ;  

    let mut epsilon = match epsilon {
        Option::Some(epsilon) => epsilon,
        Option::None => zero
    };

    let mut axis: Array<usize> = array![];
    let mut i: usize = 2;
    loop {
        if (i >= dim_x) {
            break;
        }
        axis.append(i);
        i += 1;
    };

    let mut noop_with_empty_axes = Option::Some((false));
    let mut axis_input = Option::Some(axis.span());

    if axis.len() == 0 {
        axis_input = Option::None(());
        noop_with_empty_axes = Option::Some((true));
    }

    let mut mean = self.reduce_mean( axes: axis_input, 
    keepdims: Option::Some((true)),
    noop_with_empty_axes: noop_with_empty_axes );


   
    let x_diff =  (*self) - mean;
    let x_diff_squared = x_diff * x_diff;
    let mut variance = x_diff_squared.reduce_mean(axes: axis_input,
                                keepdims: Option::Some((true)),
                                noop_with_empty_axes: noop_with_empty_axes);

    let mut dim_ones: Array<usize> = array![];
    let mut i: usize = 0;
    loop {
        if i >= dim_x - 2 {
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
    // let new_bias_shape = new_scale_shape.clone(); 
    let mut bias = bias.reshape(target_shape: new_scale_shape.span()); // since bias and scale have same shape


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

    let mut zero_vals_in_tensor = false;
    let mut i: usize = 0;
    loop {
        if (i >= std.data.len()) {
            break;
        }
        if *std.data.at(i) == zero {
            zero_vals_in_tensor = true;
        }
        i += 1;
    };

    if zero_vals_in_tensor == true {
        // clip values to min_std_val to avoid possible division by zero errors
        let mut a: usize = 500;
        let mut min_std_val = NumberTrait::<T, MAG>::half() / NumberTrait::<T, MAG>::new_unscaled(a.into(), false); 
        std = std.clip(min: Option::Some((min_std_val)), max: Option::None(()), );
    };

    let mut x_normalized = x_diff / std;

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
