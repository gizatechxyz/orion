use core::array::ArrayTrait;
use core::array::SpanTrait;
use orion::numbers::FP16x16;

use orion::operators::tensor::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;
use orion::operators::tensor::{I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FP32x32, FP32x32Impl, FixedTrait};
use orion::operators::nn::{NNTrait, FP16x16NN};


#[derive(Destruct)]
struct LinearClassifier<T> {
    classlabels: Option<Span<usize>>,
    coefficients: Span<T>,
    intercepts: Option<Span<T>>,
    multi_class: usize,
    post_transform: POST_TRANSFORM,
}


#[derive(Copy, Drop)]
enum POST_TRANSFORM {
    NONE,
    SOFTMAX,
    LOGISTIC,
    SOFTMAXZERO,
    PROBIT,
}

/// Trait
///
/// predict - Performs the linear classification.
trait LinearClassifierTrait<T> {
    /// # LinearClassifierTrait::predict
    ///
    /// ```rust 
    ///    fn predict(ref self: LinearClassifier<T>, X: Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Linear Classifier. Performs the linear classification.
    /// 
    /// ## Args
    ///
    /// * `self`: LinearClassifier<T> - A LinearClassifier object.
    /// * `X`:  Input 2D tensor.
    ///
    /// ## Returns
    ///
    /// * Tensor<T> containing the linear classification evaluation of the input X.
    ///
    /// ## Type Constraints
    ///
    /// `LinearClassifier` and `X` must be fixed points
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use orion::numbers::FP16x16;
    /// use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
    /// 
    /// use orion::operators::ml::linear::linear_classifier::{
    ///     LinearClassifierTrait, POST_TRANSFORM, LinearClassifier
    /// };
    /// 
    /// fn linear_classifier_helper(
    ///     post_transform: POST_TRANSFORM
    /// ) -> (LinearClassifier<FP16x16>, Tensor<FP16x16>) {
    /// 
    ///     let classlabels: Span<usize> = array![0, 1, 2].span();
    ///     let classlabels = Option::Some(classlabels);
    /// 
    ///     let classlabels_strings: Option<Span<FP16x16>> = Option::None;
    /// 
    ///     let coefficients: Span<FP16x16> = array![
    ///         FP16x16 { mag: 38011, sign: true },
    ///         FP16x16 { mag: 19005, sign: true },
    ///         FP16x16 { mag: 5898, sign: true },
    ///         FP16x16 { mag: 38011, sign: false },
    ///         FP16x16 { mag: 19005, sign: false },
    ///         FP16x16 { mag: 5898, sign: false }, 
    ///     ]
    ///         .span();
    /// 
    ///     let intercepts: Span<FP16x16> = array![
    ///         FP16x16 { mag: 176947, sign: false },
    ///         FP16x16 { mag: 176947, sign: true },
    ///         FP16x16 { mag: 32768, sign: false },
    ///     ]
    ///         .span();
    ///     let intercepts = Option::Some(intercepts);
    /// 
    ///     let multi_class: usize = 0;
    /// 
    ///     let mut classifier: LinearClassifier<FP16x16> = LinearClassifier {
    ///         classlabels,
    ///         coefficients,
    ///         intercepts,
    ///         multi_class,
    ///         post_transform
    ///         };
    /// 
    ///     let mut X: Tensor<FP16x16> = TensorTrait::new(
    ///         array![3, 2].span(),
    ///         array![
    ///             FP16x16 { mag: 0, sign: false },
    ///             FP16x16 { mag: 65536, sign: false },
    ///             FP16x16 { mag: 131072, sign: false },
    ///             FP16x16 { mag: 196608, sign: false },
    ///             FP16x16 { mag: 262144, sign: false },
    ///             FP16x16 { mag: 327680, sign: false },
    ///         ]
    ///             .span()
    ///     );
    /// 
    ///     (classifier, X)
    /// }
    /// 
    /// fn linear_classifier_multi_softmax() -> (Span<usize>, Tensor<FP16x16>) {
    ///     let (mut classifier, X) = linear_classifier_helper(POST_TRANSFORM::SOFTMAX);
    /// 
    ///     let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);
    /// 
    ///     (labels, scores)
    /// }
    /// 
    /// >>> 
    /// ([0, 2, 2],
    ///  [
    ///     [0.852656, 0.009192, 0.138152],
    ///     [0.318722, 0.05216, 0.629118],
    ///     [0.036323, 0.090237, 0.87344]
    ///  ])
    /// ```
    fn predict(ref self: LinearClassifier<T>, X: Tensor<T>) -> (Span<usize>, Tensor<T>);
}

impl LinearClassifierImpl<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Add<T>,
    +TensorTrait<usize>,
    +TensorTrait<T>,
    +AddEq<T>,
    +Div<T>,
    +Mul<T>,
    +Add<Tensor<T>>,
    +NNTrait<T>
> of LinearClassifierTrait<T> {
    fn predict(ref self: LinearClassifier<T>, X: Tensor<T>) -> (Span<usize>, Tensor<T>) {
        let n: usize = self.coefficients.len() / *(X.shape).at(1);
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(n);
        shape.append(*(X.shape).at(1));

        let mut coefficients = TensorTrait::new(shape.span(), self.coefficients);
        let coefficients = coefficients.transpose(array![1, 0].span());

        let mut scores = X.matmul(@coefficients);
        match self.intercepts {
            Option::Some(intercepts) => {
                let mut shape = ArrayTrait::<usize>::new();
                shape.append(1);
                shape.append(intercepts.len());
                let intercepts = TensorTrait::new(shape.span(), intercepts);
                scores = TensorTrait::add(scores, intercepts);
            },
            Option::None(_) => {},
        };

        let (n_classes, classlabels) = match self.classlabels {
            Option::Some(classlabels) => { (classlabels.len(), classlabels) },
            Option::None(_) => { (0, ArrayTrait::<usize>::new().span()) },
        };
        if *coefficients.shape.at(1) == 1 && n_classes == 2 {
            let mut new_scores = ArrayTrait::new();

            loop {
                match scores.data.pop_front() {
                    Option::Some(item) => {
                        new_scores.append(NumberTrait::neg(*item));
                        new_scores.append(*item);
                    },
                    Option::None(_) => { break; },
                }
            };
            scores = TensorTrait::new(array![*scores.shape.at(0), 2].span(), new_scores.span());
        }
        // Post Transform
        scores = match self.post_transform {
            POST_TRANSFORM::NONE => { scores },
            POST_TRANSFORM::SOFTMAX => { NNTrait::softmax(@scores, 1) },
            POST_TRANSFORM::LOGISTIC => { NNTrait::sigmoid(@scores) },
            POST_TRANSFORM::SOFTMAXZERO => { NNTrait::softmax_zero(@scores, 1)},
            POST_TRANSFORM::PROBIT => core::panic_with_felt252('Probit not supported yet'),
        };

        // Labels
        let mut labels_list = ArrayTrait::new();
        if *scores.shape.at(1) > 1 {
            let mut labels = scores.argmax(1, Option::None, Option::None);
            loop {
                match labels.data.pop_front() {
                    Option::Some(i) => { labels_list.append(*classlabels[*i]); },
                    Option::None(_) => { break; }
                };
            };
        } else {
            let mut i = 0;
            match self.post_transform {
                POST_TRANSFORM::NONE => {
                    loop {
                        if i == scores.data.len() {
                            break;
                        }
                        if *scores.data.at(i) >= NumberTrait::zero() {
                            labels_list.append(*classlabels[0]);
                        } else {
                            labels_list.append(0);
                        }
                        i += 1;
                    };
                },
                POST_TRANSFORM::SOFTMAX => {
                    loop {
                        if i == scores.data.len() {
                            break;
                        }
                        if *scores.data.at(i) >= NumberTrait::half() {
                            labels_list.append(*classlabels[0]);
                        } else {
                            labels_list.append(0);
                        }
                        i += 1;
                    };
                },
                POST_TRANSFORM::LOGISTIC => {
                    loop {
                        if i == scores.data.len() {
                            break;
                        }
                        if *scores.data.at(i) >= NumberTrait::half() {
                            labels_list.append(*classlabels[0]);
                        } else {
                            labels_list.append(0);
                        }
                        i += 1;
                    };
                },
                POST_TRANSFORM::SOFTMAXZERO => {
                    loop {
                        if i == scores.data.len() {
                            break;
                        }
                        if *scores.data.at(i) >= NumberTrait::half() {
                            labels_list.append(*classlabels[0]);
                        } else {
                            labels_list.append(0);
                        }
                        i += 1;
                    };
            },
                POST_TRANSFORM::PROBIT => core::panic_with_felt252('Probit not supported yet'),
            };
        }

        (labels_list.span(), scores)
    }
}


fn max(a: usize, b: usize) -> usize {
    if a > b {
        return a;
    } else {
        return b;
    }
}

