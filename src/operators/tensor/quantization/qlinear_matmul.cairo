use orion::numbers::{NumberTrait};
use orion::operators::tensor::quantization::dequantize_linear::dequantize_linear;
use orion::operators::tensor::quantization::quantize_linear::quantize_linear;
use orion::operators::tensor::{TensorTrait, Tensor};

/// Cf: TensorTrait::qlinear_matmul docstring
fn qlinear_matmul<
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
    impl TTensorSub: Sub<Tensor<T>>,
    impl TTensorMul: Mul<Tensor<T>>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTryInto: TryInto<T, Q>,
    //impl TTensorTryInto: TryInto<Tensor<T>, Tensor<Q>>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QCopy: Copy<Q>,
    impl QDrop: Drop<Q>,
>(
    a: @Tensor<Q>,
    a_scale: @Tensor<T>,
    a_zero_point: @Tensor<T>,
    b: @Tensor<Q>,
    b_scale: @Tensor<T>,
    b_zero_point: @Tensor<T>,
    y_scale: @Tensor<T>,
    y_zero_point: @Tensor<T>,
    min: T,
    max: T
) -> Tensor<Q> {
    let a_shape = *a.shape;
    let b_shape = *b.shape;

    let a_ndim = (a_shape).len();
    let b_ndim = (b_shape).len();

    //! Case: Both tensors are max 2-dimensional
    if a_ndim <= 2 && b_ndim <= 2 {
        let mut dequantized_a = dequantize_linear(@(*a), a_scale, a_zero_point);
        let mut dequantized_b = dequantize_linear(@(*b), b_scale, b_zero_point);

        let mut x = dequantized_a.matmul(@dequantized_b);

        return quantize_linear(@x, y_scale, y_zero_point, min, max);
    }

    // (D1, D2, M, K) * (D1, D2, K, N) -> (D1, D2, M, N)
    assert(a_ndim == b_ndim, 'dim missmatch');
    let mut dequantized_a = dequantize_linear(@(*a), a_scale, a_zero_point);
    let mut dequantized_b = dequantize_linear(@(*b), b_scale, b_zero_point);
    let mut x_shape: Array<usize> = array![];
    let mut x_data: Array<T> = array![];

    assert(a_shape[a_ndim - 1] == b_shape[b_ndim - 2], 'incompatible dim for matmul');

    let m = *a_shape[a_ndim - 2];
    let k = *a_shape[a_ndim - 1];
    let n = *b_shape[b_ndim - 1];

    let mut a_shape_reduced: Array<usize> = array![];
    a_shape_reduced.append(m);
    a_shape_reduced.append(k);

    let mut b_shape_reduced: Array<usize> = array![];
    b_shape_reduced.append(k);
    b_shape_reduced.append(n);

    let mut i = 0;
    while i != stride(a_shape)
        / (m * k) {
            result_updates(
                @subtensor(@dequantized_a, i * (m * k), a_shape_reduced.span()),
                @subtensor(@dequantized_b, i * (k * n), b_shape_reduced.span()),
                ref x_data
            );
            i += 1;
        };

    x_shape(ref x_shape, a_shape, m, n);
    let x = TensorTrait::new(x_shape.span(), x_data.span());

    quantize_linear(@x, y_scale, y_zero_point, min, max)
}

fn x_shape(ref x_data: Array<usize>, mut shape: Span<usize>, m: usize, n: usize) {
    while shape
        .len() != 2 {
            match shape.pop_front() {
                Option::Some(elem) => { x_data.append(*elem); },
                Option::None => { break; }
            };
        };

    x_data.append(m);
    x_data.append(n);
}

fn stride(mut shape: Span<usize>) -> usize {
    let shape_len = shape.len();
    assert(shape_len > 0, 'shape cannot be empty');

    let mut accumulated: usize = 1;
    loop {
        match shape.pop_back() {
            Option::Some(i) => { accumulated *= *i; },
            Option::None => { break; }
        };
    };

    accumulated
}

fn subtensor<T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    x: @Tensor<T>, start: usize, shape: Span<usize>
) -> Tensor::<T> {
    let mut data = ArrayTrait::<T>::new();
    let mut stride = stride(shape);
    let mut i = 0;

    while i != stride {
        data.append(*x.data[start + i]);
        i += 1;
    };

    TensorTrait::new(shape, data.span())
}


fn result_updates<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TMul: Mul<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mat1: @Tensor<T>, mat2: @Tensor<T>, ref result_data: Array<T>
) {
    let m = *mat1.shape[0];
    let n = *mat1.shape[1];
    let p = *mat2.shape[1];

    let mat1 = *mat1.data;
    let mat2 = *mat2.data;

    let mut result_shape: Array<usize> = array![];
    result_shape.append(m);
    result_shape.append(p);

    let mut i = 0_usize;
    while i != m {
        let mut j = 0_usize;
        while j != p {
            let mut sum: T = NumberTrait::zero();
            let mut k = 0_usize;
            while k != n {
                let mat1_index = i * n + k;
                let mat2_index = k * p + j;
                sum += *mat1[mat1_index] * *mat2[mat2_index];

                k += 1;
            };

            result_data.append(sum);
            j += 1;
        };

        i += 1;
    };
}
