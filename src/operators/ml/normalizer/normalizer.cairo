use core::array::ArrayTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor};


#[derive(Copy, Drop)]
enum NORM {
    MAX,
    L1,
    L2,
}


/// predict - Returns the normalization of the input, each row of the input is normalized independently.
trait NormalizerTrait<T> {
    /// # Normalizer::predict
    ///
    /// ```rust 
    ///    fn predict(X: Tensor<T>, norm: NORM) -> Tensor<T>;
    /// ```
    ///
    /// Returns the normalized input.
    /// Tree different types of normalization can be performed and are defined as follow :
    /// MAX: $Y = \frac{X}{max(X)}$
    /// L1: $Y = \frac{X}{sum(X)}$
    /// L2: $Y = \frac{X}\sqrt{{sum(X²)}}$
    /// For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row of the batch is normalized independently.
    ///     
    /// ## Args
    ///
    /// * `X`(`@Tensor<T>`) - Input 2D tensor. 
    /// * `norm`(`NORM`) - NORM::MAX, NORM::L1 or NORM::L2
    ///
    ///
    /// ## Returns
    ///
    /// * Tensor<T> - output tensor 
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::FP16x16;
    /// use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, FP16x16TensorDiv, FP16x16TensorPartialEq};
    /// 
    /// use orion::operators::ml::normalizer::normalizer::{
    ///     NormalizerTrait, NORM
    /// };
    /// 
    /// 
    /// 
    /// fn normalizer_max() ->  Tensor<FP16x16> {
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(3);
    ///     shape.append(3);
    ///
    ///   let mut data = ArrayTrait::new();
    ///     data.append(FP16x16 { mag: 65536, sign: true });
    ///     data.append(FP16x16 { mag: 52428, sign: true });
    ///     data.append(FP16x16 { mag: 39321, sign: true });
    ///     data.append(FP16x16 { mag: 26214, sign: true });
    ///     data.append(FP16x16 { mag: 13107, sign: true });
    ///     data.append(FP16x16 { mag: 0, sign: false });
    ///     data.append(FP16x16 { mag: 13107, sign: false });
    ///     data.append(FP16x16 { mag: 26214, sign: false });
    ///     data.append(FP16x16 { mag: 39321, sign: false });
    ///
    ///   let X = TensorTrait::new(shape.span(), data.span());
    ///
    ///   return NormalizerTrait::predict(X, NORM::MAX);
    /// }
    /// >>> [[-1.        -0.8       -0.6      ]
    ///      [-1.        -0.5        0.       ]
    ///      [ 0.3333333  0.6666666  1.       ]]
    ///
    /// ```
    ///
    ///
    fn predict(X: Tensor<T>, norm: NORM) -> Tensor<T>;
}


impl NormalizerImpl<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +TensorTrait<T>,
    +AddEq<T>,
    +Div<Tensor<T>>,
    +Mul<T>
> of NormalizerTrait<T> {
    fn predict(X: Tensor<T>, norm: NORM) -> Tensor<T> {
        assert(X.shape.len() == 2, 'input should be 2D: NxC');

        let normalized_tensor = match norm {
            NORM::MAX => { norm_max(X) },
            NORM::L1 => { norm_l1(X) },
            NORM::L2 => { norm_l2(X) },
        };

        return normalized_tensor;
    }
}


fn norm_max<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +PartialOrd<T>,
    +Div<Tensor<T>>,
>(
    X: Tensor<T>
) -> Tensor<T> {
    let div_data = reduce_max_2D_axis_1(X.abs());

    let div = TensorTrait::new(
        array![*X.shape.at(0), (div_data.len() / *X.shape.at(0))].span(), div_data
    );

    let epsillon = TensorTrait::new(array![1, 1].span(), array![NumberTrait::from_felt(1)].span());
    let safe_div = TensorTrait::max(tensors: array![div, epsillon].span());

    return X / safe_div;
}

fn norm_l1<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +AddEq<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +PartialOrd<T>,
    +Div<Tensor<T>>,
>(
    X: Tensor<T>
) -> Tensor<T> {
    let div_data = reduce_sum_2D_axis_1(X.abs());

    let div = TensorTrait::new(
        array![*X.shape.at(0), (div_data.len() / *X.shape.at(0))].span(), div_data
    );

    let epsillon = TensorTrait::new(array![1, 1].span(), array![NumberTrait::from_felt(1)].span());
    let safe_div = TensorTrait::max(tensors: array![div, epsillon].span());

    return X / safe_div;
}

fn norm_l2<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +AddEq<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +PartialOrd<T>,
    +Div<Tensor<T>>,
    +Mul<T>
>(
    X: Tensor<T>
) -> Tensor<T> {
    let div_data = reduce_sum_2D_axis_1(square(X));
    let div = TensorTrait::new(
        array![*X.shape.at(0), (div_data.len() / *X.shape.at(0))].span(), div_data
    );

    let epsillon = TensorTrait::new(array![1, 1].span(), array![NumberTrait::from_felt(1)].span());
    let safe_div = TensorTrait::max(tensors: array![div.sqrt(), epsillon].span());

    return X / safe_div;
}


fn reduce_max_2D_axis_1<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TensorTrait<T>, +PartialOrd<T>,
>(
    X: Tensor<T>
) -> Span<T> {
    let mut new_data = ArrayTrait::new();
    let N = *X.shape.at(0);
    let C = *X.shape.at(1);

    let mut i = 0;
    while i != N {
        let max = max(SpanTrait::slice(X.data, i * C, C));
        new_data.append(max);
        i += 1;

    };
    return new_data.span();
}


fn max<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TensorTrait<T>, +PartialOrd<T>,>(
    mut a: Span<T>
) -> T {
    assert(a.len() > 0, 'span cannot be empty');

    let mut max = *a.at(0);
    loop {
        match a.pop_front() {
            Option::Some(v) => { if *v > max {
                max = *v;
            }; },
            Option::None => { break max; }
        };
    }
}

fn sum<T, MAG, +Drop<T>, +Copy<T>, +AddEq<T>, +NumberTrait<T, MAG>,>(mut a: Span<T>) -> T {
    assert(a.len() > 0, 'span cannot be empty');

    let mut sum = NumberTrait::zero();
    loop {
        match a.pop_front() {
            Option::Some(v) => { sum += *v; },
            Option::None => { break sum; }
        };
    }
}

fn square<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +AddEq<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +PartialOrd<T>,
    +Mul<T>
>(
    mut a: Tensor<T>
) -> Tensor<T> {
    let mut arr = ArrayTrait::new();
    loop {
        match a.data.pop_front() {
            Option::Some(v) => { arr.append(*v * *v); },
            Option::None => { break TensorTrait::new(a.shape, arr.span()); }
        };
    }
}

fn reduce_sum_2D_axis_1<
    T, MAG, +Drop<T>, +Copy<T>, +AddEq<T>, +NumberTrait<T, MAG>, +TensorTrait<T>,
>(
    X: Tensor<T>
) -> Span<T> {
    let mut new_data = ArrayTrait::new();
    let N = *X.shape.at(0);
    let C = *X.shape.at(1);

    let mut i = 0;
    while i != N {
        let sum = sum(SpanTrait::slice(X.data, i * C, C));
        new_data.append(sum);
        i += 1;
    };
    return new_data.span();
}
