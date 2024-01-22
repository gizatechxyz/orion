use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;
use core::debug::PrintTrait;
use orion::numbers::{NumberTrait, i8};
use orion::operators::tensor::quantization::dequantize_linear::dequantize_linear;
use orion::operators::tensor::quantization::quantize_linear::quantize_linear;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::saturate;

fn dynamic_quantize_linear<
    T,
    Q,
    impl TTensor: TensorTrait<T>,
    impl QTensor: TensorTrait<Q>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TDiv: Div<T>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TTryInto: TryInto<T, Q>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QCopy: Copy<Q>,
    impl QDrop: Drop<Q>,
>(
    x: @Tensor<T>, min: T, max: T, zero: T, one: T
) -> (Tensor<Q>, Tensor<T>, Tensor<T>) {
    
    // y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
    let mut x_max: T = x.max_in_tensor();
    let mut x_min: T = x.min_in_tensor();
    if x_max < zero {
        x_max = zero;
    }
    if x_min > zero {
        x_min = zero
    }

    // scale = max == min ? 1.0f : (max - min) / float(qmax - qmin);
    let mut y_scale_values = ArrayTrait::new();
    let y_scale_value: T = (x_max - x_min) / (max - min);
    if x_max == x_min {
        y_scale_values.append(one);
    }else{
        y_scale_values.append(y_scale_value);
    }
    
    let mut y_scale_tensor_shape = ArrayTrait::new();
    y_scale_tensor_shape.append(y_scale_values.len());

    let y_scale = TensorTrait::<T>::new(
        shape: y_scale_tensor_shape.span(), data: y_scale_values.span(),
    );

    // intermediate_zero_point = qmin - min(x)/y_scale
    let intermediate_zero_point: T = min - x_min / y_scale_value;

    // y_zero_point = cast(round(saturate(itermediate_zero_point)))
    let mut y_zero_point_value: T = saturate(min, max, intermediate_zero_point);
    let mut y_zero_point_values = ArrayTrait::new();
    y_zero_point_values.append(y_zero_point_value);

    let mut y_zero_point_tensor_shape = ArrayTrait::new();
    y_zero_point_tensor_shape.append(y_zero_point_values.len());

    let mut y_zero_point_values = ArrayTrait::new();
    y_zero_point_values.append(y_zero_point_value);
    let mut y_zero_point = TensorTrait::<T>::new(
        shape: y_zero_point_tensor_shape.span(), data: y_zero_point_values.span(),
    );
    // y_zero_point = y_zero_point.round(); // tensor<FP> only supported!

    // y = saturate (round (x / y_scale) + y_zero_point)

    return (quantize_linear(x, @y_scale, @y_zero_point, min, max), y_scale, y_zero_point);
}