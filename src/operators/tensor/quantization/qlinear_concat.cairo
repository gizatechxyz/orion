use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;

use orion::numbers::{NumberTrait};
use orion::operators::tensor::quantization::dequantize_linear::dequantize_linear;
use orion::operators::tensor::quantization::quantize_linear::quantize_linear;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::math::concat::{
    validate_shapes, compute_output_size, concatenate_data
};

fn qlinear_concat<
    T,
    MAG,
    Q,
    impl TTensor: TensorTrait<T>,
    impl QTensor: TensorTrait<Q>,
    impl QIntoT: Into<Q, T>,
    impl QTensorIntoTTensor: Into<Tensor<Q>, Tensor<T>>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TDiv: Div<T>,
    impl TMul: Mul<T>,
    impl TTensorAdd: Add<Tensor<T>>,
    impl TTensorSub: Sub<Tensor<T>>,
    impl TTensorMul: Mul<Tensor<T>>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTryInto: TryInto<T, Q>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QCopy: Copy<Q>,
    impl QDrop: Drop<Q>,
>(
    tensors: Span<Tensor<Q>>,
    scales: Span<Tensor<T>>,
    zero_points: Span<Tensor<T>>,
    y_scale: @Tensor<T>,
    y_zero_point: @Tensor<T>,
    axis: usize,
    min: T,
    max: T
) -> Tensor<Q> {
    assert(tensors.len() == scales.len(), 'Each Tensors must have a scale');
    assert(tensors.len() == zero_points.len(), 'Each Tensors must have a scale');

    //let mut x = TensorTrait::concat(tensors: array![dequantized_a, dequantized_b].span(), axis: axis);
    let mut x = concat_dequantize(tensors, scales, zero_points, axis, min, max);

    return quantize_linear(@x, y_scale, y_zero_point, min, max);
}


fn concat_dequantize<
    T,
    MAG,
    Q,
    impl TTensor: TensorTrait<T>,
    impl QTensor: TensorTrait<Q>,
    impl QIntoT: Into<Q, T>,
    impl QTensorIntoTTensor: Into<Tensor<Q>, Tensor<T>>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TTensorSub: Sub<Tensor<T>>,
    impl TTensorMul: Mul<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTryInto: TryInto<T, Q>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QCopy: Copy<Q>,
    impl QDrop: Drop<Q>,
>(
    tensors: Span<Tensor<Q>>,
    scales: Span<Tensor<T>>,
    zero_points: Span<Tensor<T>>,
    axis: usize,
    min: T,
    max: T
) -> Tensor<T> {
    assert(tensors.len() >= 2, 'Input tensors must be > 1');
    let base_tensor = *tensors.at(0);
    let base_shape = base_tensor.shape;
    let dimension = base_shape.len();
    assert(dimension > axis, 'Out of bounds for dimension');

    // Validate shapes of tensors
    validate_shapes(tensors, base_shape, axis);

    // Calculate output size
    let output_size = compute_output_size(base_shape, tensors, axis);

    // Dequantize tensors
    let tensors = dequantize_tensors(tensors, scales, zero_points, min, max);

    // Concatenate tensor data
    let output_data: Array<T> = concatenate_data(tensors, axis, base_shape);

    TensorTrait::<T>::new(output_size.span(), output_data.span())
}

fn dequantize_tensors<
    Q,
    T,
    impl TTensor: TensorTrait<T>,
    impl QIntoT: Into<Q, T>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TTensorSub: Sub<Tensor<T>>,
    impl TTensorMul: Mul<Tensor<T>>,
    impl QTensorIntoTTensor: Into<Tensor<Q>, Tensor<T>>,
    impl TDrop: Drop<T>,
    impl TCopy: Copy<T>,
    impl QCopy: Copy<Q>,
    impl QDrop: Drop<Q>
//MAybe numberTRait
>(
    mut tensors: Span<Tensor<Q>>,
    scales: Span<Tensor<T>>,
    zero_points: Span<Tensor<T>>,
    min: T,
    max: T
) -> Span<Tensor<T>> {
    let mut array = ArrayTrait::<Tensor<T>>::new();
    let mut i = 0;
    loop {
        match tensors.pop_front() {
            Option::Some(tensor) => {
                array
                    .append(dequantize_linear(@(*tensor), @(*scales.at(i)), @(*zero_points.at(i))));
            },
            Option::None(_) => { break; }
        };
        i += 1;
    };
    return array.span();
}
/// # tensor.concat
///
/// ```rust 
///    fn concat(tensors: Span<Tensor<T>>, axis: usize,  ) -> Tensor<T>;
/// ```
///
/// Concatenate a list of tensors into a single tensor.
///
/// ## Args
///
/// * `tensors`(` Span<Tensor<T>>,`) - Array of the input tensors.
/// * `axis`(`usize`) -  Axis to concat on.
///
/// ## Panics
///
/// * Panic if tensor length is not greater than 1.
/// * Panics if dimension is not greater than axis.
///
/// ## Returns 
///
/// A new `Tensor<T>` concatenated tensor of the input tensors.
///
/// ## Example
///
/// ```rust
/// use core::array::{ArrayTrait, SpanTrait};
/// 
/// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
/// 
/// fn concat_example() -> Tensor<u32> {
///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
///     let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
///     let result = TensorTrait::concat(tensors: array![tensor1, tensor2].span(), axis: 0);
///     return result;
/// }
/// >>> [[0. 1.]
///      [2. 3.],
///      [0. 1.]
///      [2. 3.]]
///
///     result.shape
/// >>> (4, 2)
///
///    let result = TensorTrait::concat(tensors: array![tensor1, tensor2].span(), axis: 1);
///    return result;
/// }
/// >>> [[0. 1., 0., 1.]
///      [2. 3., 2., 3.]]
///
///     result.shape
/// >>> (2, 4 ) 
/// ```
///
///fn concat(tensors: Span<Tensor<T>>, axis: usize,) -> Tensor<T>;


