use core::array::ArrayTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::{
    TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor, BoolTensor
};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};

use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};

use orion::numbers::{FP64x64, FP64x64Impl};
use orion::operators::tensor::implementations::tensor_fp64x64::{FP64x64Tensor};
use orion::operators::nn::{NNTrait, FP16x16NN, FP64x64NN};
use orion::utils::get_row;

use orion::operators::ml::svm::core::{kernel_dot, KERNEL_TYPE};
use orion::operators::ml::POST_TRANSFORM;


#[derive(Copy, Drop, Destruct)]
struct SVMClassifier<T> {
    classlabels: Span<usize>,
    coefficients: Span<T>,
    kernel_params: Span<T>,
    kernel_type: KERNEL_TYPE,
    post_transform: POST_TRANSFORM,
    prob_a: Span<T>,
    prob_b: Span<T>,
    rho: Span<T>,
    support_vectors: Span<T>,
    vectors_per_class: Option<Span<usize>>,
}

#[derive(Copy, Drop)]
enum MODE {
    SVM_LINEAR,
    SVM_SVC,
}


///
/// predict - Returns the top class for each of N inputs.
trait SVMClassifierTrait<T> {
    /// # SVMClassifierTrait::predict
    ///
    /// ```rust 
    ///    fn predict(ref self: SVMClassifier<T>, X: Tensor<T>) -> (Span<usize>, Tensor<T>);
    /// ```
    ///
    /// Support Vector Machine classification.
    /// 
    /// ## Args
    ///
    /// * `self`: SVMClassifier<T> - A SVMClassifier object.
    /// * `X`:  Input 2D tensor.
    ///
    /// ## Returns
    ///
    /// * N Top class for each point
    /// * The class score Matrix for each class, for each point.  If prob_a and prob_b are provided they are probabilities for each class, otherwise they are raw scores.
    ///
    /// ## Type Constraints
    ///
    /// `SVMClassifier` and `X` must be fixed points
    ///
    /// ## Examples
    /// 
    /// ```rust    
    /// fn example_svm_classifier_noprob_linear_sv_none() -> (Span<usize>, Tensor<FP16x16>) {
    ///     let coefficients: Span<FP16x16> = array![
    ///         FP16x16 { mag: 50226, sign: false },
    ///         FP16x16 { mag: 5711, sign: false },
    ///         FP16x16 { mag: 7236, sign: false },
    ///         FP16x16 { mag: 63175, sign: true }
    ///     ]
    ///         .span();
    ///     let kernel_params: Span<FP16x16> = array![
    ///         FP16x16 { mag: 8025, sign: false },
    ///         FP16x16 { mag: 0, sign: false },
    ///         FP16x16 { mag: 196608, sign: false }
    ///     ]
    ///         .span();
    ///     let kernel_type = KERNEL_TYPE::LINEAR;
    ///     let prob_a: Span<FP16x16> = array![].span();
    ///     let prob_b: Span<FP16x16> = array![].span();
    ///     let rho: Span<FP16x16> = array![FP16x16 { mag: 146479, sign: false }].span();
    /// 
    ///     let support_vectors: Span<FP16x16> = array![
    ///         FP16x16 { mag: 314572, sign: false },
    ///         FP16x16 { mag: 222822, sign: false },
    ///         FP16x16 { mag: 124518, sign: false },
    ///         FP16x16 { mag: 327680, sign: false },
    ///         FP16x16 { mag: 196608, sign: false },
    ///         FP16x16 { mag: 104857, sign: false },
    ///         FP16x16 { mag: 294912, sign: false },
    ///         FP16x16 { mag: 150732, sign: false },
    ///         FP16x16 { mag: 85196, sign: false },
    ///         FP16x16 { mag: 334233, sign: false },
    ///         FP16x16 { mag: 163840, sign: false },
    ///         FP16x16 { mag: 196608, sign: false }
    ///     ]
    ///         .span();
    ///     let classlabels: Span<usize> = array![0, 1].span();
    /// 
    ///     let vectors_per_class = Option::Some(array![3, 1].span());
    /// 
    ///     let post_transform = POST_TRANSFORM::NONE;
    /// 
    ///     let mut classifier: SVMClassifier<FP16x16> = SVMClassifier {
    ///         classlabels,
    ///         coefficients,
    ///         kernel_params,
    ///         kernel_type,
    ///         post_transform,
    ///         prob_a,
    ///         prob_b,
    ///         rho,
    ///         support_vectors,
    ///         vectors_per_class,
    ///     };
    /// 
    ///     let mut X: Tensor<FP16x16> = TensorTrait::new(
    ///         array![3, 3].span(),
    ///         array![
    ///             FP16x16 { mag: 65536, sign: true },
    ///             FP16x16 { mag: 52428, sign: true },
    ///             FP16x16 { mag: 39321, sign: true },
    ///             FP16x16 { mag: 26214, sign: true },
    ///             FP16x16 { mag: 13107, sign: true },
    ///             FP16x16 { mag: 0, sign: false },
    ///             FP16x16 { mag: 13107, sign: false },
    ///             FP16x16 { mag: 26214, sign: false },
    ///             FP16x16 { mag: 39321, sign: false },
    ///         ]
    ///             .span()
    ///     );
    /// 
    ///     return SVMClassifierTrait::predict(ref classifier, X);
    /// 
    /// }
    /// // >>> ([0, 0, 0],
    /// //      [[-2.662655, 2.662655], 
    /// //       [-2.21481, 2.21481], 
    /// //       [-1.766964, 1.766964]])
    /// 
    /// 
    /// fn example_svm_classifier_binary_softmax_fp64x64() -> (Span<usize>, Tensor<FP64x64>) {
    ///     let coefficients: Span<FP64x64> = array![
    ///         FP64x64 { mag: 18446744073709551616, sign: false },
    ///         FP64x64 { mag: 18446744073709551616, sign: false },
    ///         FP64x64 { mag: 18446744073709551616, sign: false },
    ///         FP64x64 { mag: 18446744073709551616, sign: false },
    ///         FP64x64 { mag: 18446744073709551616, sign: true },
    ///         FP64x64 { mag: 18446744073709551616, sign: true },
    ///         FP64x64 { mag: 18446744073709551616, sign: true },
    ///         FP64x64 { mag: 18446744073709551616, sign: true }
    ///     ]
    ///         .span();
    ///     let kernel_params: Span<FP64x64> = array![
    ///         FP64x64 { mag: 7054933896252620800, sign: false },
    ///         FP64x64 { mag: 0, sign: false },
    ///         FP64x64 { mag: 55340232221128654848, sign: false }
    ///     ]
    ///         .span();
    ///     let kernel_type = KERNEL_TYPE::RBF;
    ///     let prob_a: Span<FP64x64> = array![FP64x64 { mag: 94799998099962986496, sign: true }].span();
    ///     let prob_b: Span<FP64x64> = array![FP64x64 { mag: 1180576833385529344, sign: false }].span();
    ///     let rho: Span<FP64x64> = array![FP64x64 { mag: 3082192501545631744, sign: false }].span();
    /// 
    ///     let support_vectors: Span<FP64x64> = array![
    ///         FP64x64 { mag: 3528081300248330240, sign: false },
    ///         FP64x64 { mag: 19594207602596118528, sign: true },
    ///         FP64x64 { mag: 9235613999318433792, sign: false },
    ///         FP64x64 { mag: 10869715877100519424, sign: true },
    ///         FP64x64 { mag: 5897111318564962304, sign: true },
    ///         FP64x64 { mag: 1816720038917308416, sign: false },
    ///         FP64x64 { mag: 4564890528671334400, sign: false },
    ///         FP64x64 { mag: 21278987070814027776, sign: true },
    ///         FP64x64 { mag: 7581529597213147136, sign: false },
    ///         FP64x64 { mag: 10953113834067329024, sign: true },
    ///         FP64x64 { mag: 24318984989010034688, sign: true },
    ///         FP64x64 { mag: 30296187483321270272, sign: true },
    ///         FP64x64 { mag: 10305112258191032320, sign: false },
    ///         FP64x64 { mag: 17005441559857987584, sign: true },
    ///         FP64x64 { mag: 11555205301925838848, sign: false },
    ///         FP64x64 { mag: 2962701975885447168, sign: true },
    ///         FP64x64 { mag: 11741665981322231808, sign: true },
    ///         FP64x64 { mag: 15376232508819505152, sign: false },
    ///         FP64x64 { mag: 13908474645692022784, sign: false },
    ///         FP64x64 { mag: 7323415394302033920, sign: true },
    ///         FP64x64 { mag: 3284258824352956416, sign: true },
    ///         FP64x64 { mag: 11374683084831064064, sign: true },
    ///         FP64x64 { mag: 9087138148126818304, sign: false },
    ///         FP64x64 { mag: 8247488946750095360, sign: false }
    ///     ]
    ///         .span();
    ///     let classlabels: Span<usize> = array![0, 1].span();
    /// 
    ///     let vectors_per_class = Option::Some(array![4, 4].span());
    ///     let post_transform = POST_TRANSFORM::SOFTMAX;
    /// 
    ///     let mut classifier: SVMClassifier<FP64x64> = SVMClassifier {
    ///         classlabels,
    ///         coefficients,
    ///         kernel_params,
    ///         kernel_type,
    ///         post_transform,
    ///         prob_a,
    ///         prob_b,
    ///         rho,
    ///         support_vectors,
    ///         vectors_per_class,
    ///     };
    /// 
    ///     let mut X: Tensor<FP64x64> = TensorTrait::new(
    ///         array![3, 3].span(),
    ///         array![
    ///             FP64x64 { mag: 18446744073709551616, sign: true },
    ///             FP64x64 { mag: 14757395258967642112, sign: true },
    ///             FP64x64 { mag: 11068046444225730560, sign: true },
    ///             FP64x64 { mag: 7378697629483821056, sign: true },
    ///             FP64x64 { mag: 3689348814741910528, sign: true },
    ///             FP64x64 { mag: 0, sign: false },
    ///             FP64x64 { mag: 3689348814741910528, sign: false },
    ///             FP64x64 { mag: 7378697629483821056, sign: false },
    ///             FP64x64 { mag: 11068046444225730560, sign: false }
    ///         ]
    ///             .span()
    ///     );
    /// 
    /// 
    ///     return SVMClassifierTrait::predict(ref classifier, X);
    /// 
    /// }
    /// >>> ([0, 1, 1],
    ///      [[0.728411, 0.271589], 
    ///       [0.484705, 0.515295], 
    ///       [0.274879, 0.725121]])
    /// ```
    fn predict(ref self: SVMClassifier<T>, X: Tensor<T>) -> (Span<usize>, Tensor<T>);
}

impl SVMClassifierImpl<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +Into<usize, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Add<T>,
    +TensorTrait<T>,
    +AddEq<T>,
    +Div<T>,
    +Mul<T>,
    +Neg<T>,
    +Sub<T>,
    +NNTrait<T>,
> of SVMClassifierTrait<T> {
    fn predict(ref self: SVMClassifier<T>, X: Tensor<T>) -> (Span<usize>, Tensor<T>) {
        let mut vector_count_ = 0;
        let class_count_ = max(self.classlabels.len(), 1);
        let mut starting_vector_: Array<usize> = array![];

        let (vectors_per_class_, starting_vector_) = match self.vectors_per_class {
            Option::Some(vectors_per_class) => {
                let mut i = 0;
                while i != vectors_per_class
                    .len() {
                        starting_vector_.append(vector_count_);
                        vector_count_ += *vectors_per_class.at(i);
                        i += 1;
                    };

                (vectors_per_class, starting_vector_.span())
            },
            Option::None => { (array![].span(), array![].span()) },
        };

        let (mode, kernel_type_, sv, coefs) = if vector_count_ > 0 {
            let mode = MODE::SVM_SVC;
            let kernel_type_ = self.kernel_type;
            let sv = TensorTrait::new(
                array![vector_count_, self.support_vectors.len() / vector_count_].span(),
                self.support_vectors
            );
            let coefs = TensorTrait::new(
                array![self.coefficients.len() / vector_count_, vector_count_].span(),
                self.coefficients
            );
            (mode, kernel_type_, sv, coefs)
        } else {
            let mode = MODE::SVM_LINEAR;
            let kernel_type_ = KERNEL_TYPE::LINEAR;
            let sv = TensorTrait::new(
                array![self.support_vectors.len()].span(), self.support_vectors
            );
            let coefs = TensorTrait::new(
                array![class_count_, self.coefficients.len() / class_count_].span(),
                self.coefficients
            );
            (mode, kernel_type_, sv, coefs)
        };

        let weights_are_all_positive_ = (min(self.coefficients) >= NumberTrait::zero());

        // SVM
        let (res, votes) = match mode {
            MODE::SVM_LINEAR => {
                let mut res: Array<T> = array![];
                let mut n = 0;
                while n != *X
                    .shape
                    .at(0) {
                        let mut x_n = get_row(@X, n);
                        let scores = run_linear(ref self, x_n, coefs, class_count_, kernel_type_);
                        let mut i = 0;
                        while i != scores.len() {
                            res.append(*scores.at(i));
                            i += 1;
                        };

                        n += 1;
                    };

                (
                    TensorTrait::new(array![*X.shape.at(0), class_count_].span(), res.span()),
                    Option::None
                )
            },
            MODE::SVM_SVC => {
                let mut res: Array<T> = array![];
                let mut votes: Array<T> = array![];
                let mut n = 0;
                while n != *X
                    .shape
                    .at(0) {
                        let mut x_n = get_row(@X, n);
                        let (scores, mut vote) = run_svm(
                            ref self,
                            x_n,
                            sv,
                            vector_count_,
                            kernel_type_,
                            class_count_,
                            starting_vector_,
                            coefs,
                            vectors_per_class_
                        );
                        let mut i = 0;
                        while i != scores.len() {
                            res.append(*scores.at(i));
                            i += 1;
                        };

                        let mut i = 0;
                        while i != vote.len() {
                            votes.append(vote.at(i));
                            i += 1;
                        };

                        n += 1;
                    };

                (
                    TensorTrait::new(
                        array![*X.shape.at(0), class_count_ * (class_count_ - 1) / 2].span(),
                        res.span()
                    ),
                    Option::Some(
                        TensorTrait::new(array![*X.shape.at(0), class_count_].span(), votes.span())
                    )
                )
            },
        };

        // Proba
        let (scores, has_proba) = match mode {
            MODE::SVM_LINEAR => { (res, false) },
            MODE::SVM_SVC => {
                let (scores, has_proba) = if self.prob_a.len() > 0 {
                    let mut scores: Array<T> = array![];
                    let mut n = 0;
                    while n != *res
                        .shape
                        .at(0) {
                            let res_n = get_row(@res, n);
                            let mut s = probablities(ref self, res_n, class_count_);

                            let mut i = 0;
                            while i != s.len() {
                                scores.append(s.at(i));
                                i += 1;
                            };

                            n += 1;
                        };
                    (
                        TensorTrait::new(
                            array![*res.shape.at(0), scores.len() / *res.shape.at(0)].span(),
                            scores.span()
                        ),
                        true
                    )
                } else {
                    (res, false)
                };

                (scores, has_proba)
            },
        };

        // Finalization 
        let mut labels: Array<usize> = array![];
        let mut final_scores: Array<T> = array![];

        let mut n = 0;
        while n != *scores
            .shape
            .at(0) {
                let mut scores_n = get_row(@scores, n);
                match votes {
                    Option::Some(votes) => {
                        let mut votes_n = get_row(@votes, n);
                        let (label, new_scores) = compute_final_scores(
                            ref self,
                            votes_n,
                            scores_n,
                            weights_are_all_positive_,
                            has_proba,
                            self.classlabels
                        );

                        let mut i = 0;
                        while i != new_scores
                            .data
                            .len() {
                                final_scores.append(*new_scores.data.at(i));
                                i += 1;
                            };

                        labels.append(label);
                    },
                    Option::None => {
                        let (label, new_scores) = compute_final_scores(
                            ref self,
                            array![].span(),
                            scores_n,
                            weights_are_all_positive_,
                            has_proba,
                            self.classlabels
                        );

                        let mut i = 0;
                        while i != new_scores
                            .data
                            .len() {
                                final_scores.append(*new_scores.data.at(i));
                                i += 1;
                            };

                        labels.append(label);
                    },
                }

                n += 1;
            };

        let labels = labels.span();

        // Labels
        if self.classlabels.len() > 0 {
            let mut class_labels: Array<usize> = array![];
            let mut i = 0;
            while i != labels
                .len() {
                    class_labels.append(*self.classlabels.at(*labels.at(i)));
                    i += 1;
                };

            return (
                class_labels.span(),
                TensorTrait::new(
                    array![*X.shape.at(0), final_scores.len() / *X.shape.at(0)].span(),
                    final_scores.span()
                )
            );
        }

        (
            labels,
            TensorTrait::new(
                array![*X.shape.at(0), final_scores.len() / *X.shape.at(0)].span(),
                final_scores.span()
            )
        )
    }
}

fn run_svm<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +Add<T>,
    +TensorTrait<T>,
    +AddEq<T>,
    +Mul<T>,
    +Neg<T>,
    +Sub<T>,
    +PartialOrd<T>,
>(
    ref self: SVMClassifier<T>,
    X: Span<T>,
    sv: Tensor<T>,
    vector_count_: usize,
    kernel: KERNEL_TYPE,
    class_count_: usize,
    starting_vector_: Span<usize>,
    coefs: Tensor<T>,
    vectors_per_class_: Span<usize>
) -> (Array<T>, NullableVec<T>) {
    let mut evals = 0;
    let mut kernels: Array<T> = array![];

    let mut j = 0;
    while j != vector_count_ {
        let sv_j = get_row(@sv, j);
        kernels.append(kernel_dot(self.kernel_params, X, sv_j, kernel));
        j += 1;
    };

    let kernels = kernels.span();

    let mut scores: Array<T> = array![];
    let mut votes = VecTrait::new();
    VecTrait::set(ref votes, class_count_ - 1, NumberTrait::zero());

    let mut i = 0;
    while i != class_count_ {
        let si_i = *starting_vector_.at(i);
        let class_i_sc = *vectors_per_class_.at(i);

        let mut j = i + 1;
        while j != class_count_ {
            let si_j = *starting_vector_.at(j);
            let class_j_sc = *vectors_per_class_.at(j);

            let s1 = dot_start_end(
                coefs.data,
                kernels,
                (j - 1) * *coefs.shape.at(0) + si_i,
                (j - 1) * *coefs.shape.at(0) + si_i + class_i_sc,
                si_i,
                si_i + class_i_sc
            );

            let s2 = dot_start_end(
                coefs.data,
                kernels,
                i * *coefs.shape.at(0) + si_j,
                i * *coefs.shape.at(0) + si_j + class_j_sc,
                si_j,
                si_j + class_j_sc
            );

            let s = *self.rho.at(evals) + s1 + s2;
            scores.append(s);

            if s > NumberTrait::zero() {
                VecTrait::set(ref votes, i, VecTrait::at(ref votes, i) + NumberTrait::one());
            } else {
                VecTrait::set(ref votes, j, VecTrait::at(ref votes, j) + NumberTrait::one());
            }

            evals += 1;
            j += 1;
        };

        i += 1;
    };

    (scores, votes)
}

fn run_linear<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +Add<T>,
    +TensorTrait<T>,
    +AddEq<T>,
    +Mul<T>,
    +Neg<T>,
    +Sub<T>,
>(
    ref self: SVMClassifier<T>,
    X: Span<T>,
    coefs: Tensor<T>,
    class_count_: usize,
    kernel: KERNEL_TYPE
) -> Array<T> {
    let mut scores: Array<T> = array![];

    let mut j = 0;
    while j != class_count_ {
        let coefs_j = get_row(@coefs, j);

        let d = kernel_dot(self.kernel_params, X, coefs_j, kernel);

        let score = *self.rho.at(0) + d;

        scores.append(score);
        j += 1;
    };

    scores
}

fn compute_final_scores<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +NNTrait<T>,
    +Into<usize, MAG>,
    +Add<T>,
    +TensorTrait<T>,
    +AddEq<T>,
    +Mul<T>,
    +Neg<T>,
    +Sub<T>,
    +Div<T>,
    +PartialOrd<T>,
>(
    ref self: SVMClassifier<T>,
    votes: Span<T>,
    scores: Span<T>,
    weights_are_all_positive_: bool,
    has_proba: bool,
    classlabels: Span<usize>
) -> (usize, Tensor<T>) {
    let (max_class, max_weight) = if votes.len() > 0 {
        let max_class = argmax_span(votes);
        let max_weight = *votes.at(max_class);
        (max_class, max_weight)
    } else {
        let max_class = argmax_span(scores);
        let max_weight = *scores.at(max_class);
        (max_class, max_weight)
    };

    let (label, write_additional_scores) = if self.rho.len() == 1 {
        let (label, write_additional_scores) = set_score_svm(
            max_weight, max_class, weights_are_all_positive_, has_proba, classlabels, 1, 0
        );
        (label, write_additional_scores)
    } else if classlabels.len() > 0 {
        let label = *classlabels.at(max_class);
        (label, 4)
    } else {
        (max_class, 4)
    };

    let new_scores = write_scores(
        scores.len(),
        TensorTrait::new(array![scores.len()].span(), scores),
        self.post_transform,
        write_additional_scores
    );

    (label, new_scores)
}

fn write_scores<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +PartialOrd<T>,
    +NNTrait<T>,
    +Neg<T>,
    +Sub<T>,
>(
    n_classes: usize, scores: Tensor<T>, post_transform: POST_TRANSFORM, add_second_class: usize
) -> Tensor<T> {
    let new_scores = if n_classes >= 2 {
        let new_scores = match post_transform {
            POST_TRANSFORM::NONE => scores,
            POST_TRANSFORM::SOFTMAX => NNTrait::softmax(@scores, Option::Some(0)),
            POST_TRANSFORM::LOGISTIC => NNTrait::sigmoid(@scores),
            POST_TRANSFORM::SOFTMAXZERO => NNTrait::softmax_zero(@scores, 0),
            POST_TRANSFORM::PROBIT => core::panic_with_felt252('Probit not supported yet'),
        };
        new_scores
    } else { //if n_classes == 1
        let new_scores = match post_transform {
            POST_TRANSFORM::NONE => {
                let scores = if add_second_class == 0 || add_second_class == 1 {
                    TensorTrait::new(
                        array![2].span(),
                        array![NumberTrait::one() - *scores.data.at(0), *scores.data.at(0)].span()
                    )
                } else if add_second_class == 2 || add_second_class == 3 {
                    TensorTrait::new(
                        array![2].span(), array![-*scores.data.at(0), *scores.data.at(0)].span()
                    )
                } else {
                    TensorTrait::new(array![1].span(), array![*scores.data.at(0)].span())
                };

                scores
            },
            POST_TRANSFORM::SOFTMAX => {
                let scores = if add_second_class == 0 || add_second_class == 1 {
                    TensorTrait::new(
                        array![2].span(),
                        array![NumberTrait::one() - *scores.data.at(0), *scores.data.at(0)].span()
                    )
                } else if add_second_class == 2 || add_second_class == 3 {
                    //
                    NNTrait::softmax(
                        @TensorTrait::new(
                            array![2].span(), array![-*scores.data.at(0), *scores.data.at(0)].span()
                        ),
                        Option::Some(0)
                    )
                } else {
                    TensorTrait::new(array![1].span(), array![*scores.data.at(0)].span())
                };

                scores
            },
            POST_TRANSFORM::LOGISTIC => {
                let scores = if add_second_class == 0 || add_second_class == 1 {
                    TensorTrait::new(
                        array![2].span(),
                        array![NumberTrait::one() - *scores.data.at(0), *scores.data.at(0)].span()
                    )
                } else if add_second_class == 2 || add_second_class == 3 {
                    //
                    NNTrait::sigmoid(
                        @TensorTrait::new(
                            array![2].span(), array![-*scores.data.at(0), *scores.data.at(0)].span()
                        )
                    )
                } else {
                    TensorTrait::new(array![1].span(), array![*scores.data.at(0)].span())
                };

                scores
            },
            POST_TRANSFORM::SOFTMAXZERO => {
                let scores = if add_second_class == 0 || add_second_class == 1 {
                    TensorTrait::new(
                        array![2].span(),
                        array![NumberTrait::one() - *scores.data.at(0), *scores.data.at(0)].span()
                    )
                } else if add_second_class == 2 || add_second_class == 3 {
                    //
                    NNTrait::softmax_zero(
                        @TensorTrait::new(
                            array![2].span(), array![-*scores.data.at(0), *scores.data.at(0)].span()
                        ),
                        0
                    )
                } else {
                    TensorTrait::new(array![1].span(), array![*scores.data.at(0)].span())
                };

                scores
            },
            POST_TRANSFORM::PROBIT => core::panic_with_felt252('Probit not applicable here.'),
        };
        new_scores
    };

    new_scores
}

fn set_score_svm<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +TensorTrait<T>, +PartialOrd<T>,
>(
    max_weight: T,
    maxclass: usize,
    weights_are_all_positive_: bool,
    has_proba: bool,
    classlabels: Span<usize>,
    posclass: usize,
    negclass: usize
) -> (usize, usize) {
    let mut write_additional_scores = 0;

    if classlabels.len() == 2 {
        write_additional_scores = 2;
        if !has_proba {
            if weights_are_all_positive_ && max_weight >= NumberTrait::half() {
                return (*classlabels.at(1), write_additional_scores);
            };
        };

        return (*classlabels.at(maxclass), write_additional_scores);
    }
    if max_weight >= NumberTrait::zero() {
        return (posclass, write_additional_scores);
    };

    (negclass, write_additional_scores)
}

fn argmax_span<T, +Drop<T>, +Copy<T>, +PartialOrd<T>,>(span: Span<T>) -> usize {
    let mut max = 0;
    let mut i = 0;
    while i != span.len() {
        if *span.at(i) > *span.at(max) {
            max = i;
        }

        i += 1;
    };

    max
}

fn probablities<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +Into<usize, MAG>,
    +Add<T>,
    +TensorTrait<T>,
    +AddEq<T>,
    +Mul<T>,
    +Neg<T>,
    +Sub<T>,
    +Div<T>,
    +PartialOrd<T>,
>(
    ref self: SVMClassifier<T>, scores: Span<T>, class_count_: usize
) -> NullableVec<T> {
    let mut probsp2: MutMatrix<T> = MutMatrixImpl::new(class_count_, class_count_);
    let mut index = 0;
    let mut i = 0;
    while i != class_count_ {
        let mut j = i + 1;
        while j != class_count_ {
            let val1 = sigmoid_probability(
                *scores.at(index), *self.prob_a.at(index), *self.prob_b.at(index)
            );

            let mut val2 = NumberTrait::min(
                val1, NumberTrait::one()
            ); // ONNX : min(val2, (1 - 1.0e-7))

            probsp2.set(i, j, val2);
            probsp2.set(j, i, NumberTrait::one() - val2);

            j += 1;
            index += 1;
        };

        i += 1;
    };

    multiclass_probability(class_count_, ref probsp2)
}

fn multiclass_probability<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +Add<T>,
    +Mul<T>,
    +Div<T>,
    +Sub<T>,
    +Neg<T>,
    +AddEq<T>,
    +Into<usize, MAG>,
>(
    k: usize, ref R: MutMatrix<T>
) -> NullableVec<T> {
    let max_iter = max(100, k);
    let k_fp = NumberTrait::<T>::new_unscaled(k.into(), false);

    let mut Q: MutMatrix<T> = MutMatrixImpl::new(k, k);
    MutMatrixImpl::set(ref Q, k - 1, k - 1, NumberTrait::zero());

    let mut P = VecTrait::new();
    VecTrait::set(ref P, k - 1, NumberTrait::zero());

    let a: usize = 100;
    let eps = (NumberTrait::half() / NumberTrait::new_unscaled(a.into(), false)) / k_fp;
    let mut t = 0;

    while t != k {
        VecTrait::set(ref P, t, NumberTrait::one() / k_fp);

        let mut i = 0;
        let mut acc1 = NumberTrait::zero();
        while i != t {
            let r_i = MutMatrixImpl::at(ref R, i, t);
            acc1 += r_i * r_i;
            i += 1;
        };

        MutMatrixImpl::set(ref Q, t, t, acc1);

        let mut i = 0;
        while i != t {
            MutMatrixImpl::set(ref Q, t, i, MutMatrixImpl::at(ref Q, i, t));
            i += 1;
        };

        let mut i = t + 1;
        let mut acc2 = NumberTrait::zero();
        while i != k {
            let r_i = MutMatrixImpl::at(ref R, i, t);
            acc2 += r_i * r_i;
            i += 1;
        };

        MutMatrixImpl::set(ref Q, t, t, acc1 + acc2);

        let mut i = t + 1;
        let mut acc = NumberTrait::zero();
        while i != k {
            acc += -MutMatrixImpl::at(ref R, i, t) * MutMatrixImpl::at(ref R, t, i);
            i += 1;
        };

        let mut i = t + 1;
        while i != k {
            MutMatrixImpl::set(ref Q, t, i, acc);
            i += 1;
        };

        t += 1;
    };

    let mut i = 0;
    while i != max_iter {
        let mut Qp = MutMatrixImpl::matrix_vector_product(ref Q, ref P);
        let mut pQp = dot(ref P, ref Qp);

        let mut max_error = NumberTrait::zero();
        let mut t = 0;
        while t != k {
            let error = NumberTrait::abs(Qp.at(t) - pQp);
            if error > max_error {
                max_error = error;
            }

            t += 1;
        };

        if max_error < eps {
            break;
        }

        let mut t = 0;
        while t != k {
            let diff = (-VecTrait::at(ref Qp, t) + pQp) / MutMatrixImpl::at(ref Q, t, t);
            VecTrait::set(ref P, t, VecTrait::at(ref P, t) + diff);

            pQp =
                (pQp
                    + diff
                        * (diff * MutMatrixImpl::at(ref Q, t, t)
                            + (NumberTrait::one() + NumberTrait::one()) * VecTrait::at(ref Qp, t)))
                / ((NumberTrait::one() + diff) * (NumberTrait::one() + diff));

            div_element_wise(ref P, NumberTrait::one() + diff);

            Qp_computation(ref Q, ref Qp, diff, t);

            t += 1;
        };

        i += 1;
    };

    P
}

/// Computation of the matrix Qb in the multiclass_probability computation
///
/// Qp[:] = (Qp + diff * Q[t, :]) / (1 + diff)
///
fn Qp_computation<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +Mul<T>,
    +Add<T>,
    +Div<T>,
    +AddEq<T>
>(
    ref Q: MutMatrix<T>, ref Qp: NullableVec<T>, diff: T, t: usize
) {
    let m = Qp.len;

    let mut i = 0_usize;
    while i != m {
        let elem = (VecTrait::at(ref Qp, i) + diff * MutMatrixImpl::at(ref Q, t, i))
            / (NumberTrait::one() + diff);

        VecTrait::set(ref Qp, i, elem);
        i += 1;
    };
}

fn sigmoid_probability<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +Add<T>,
    +Mul<T>,
    +Div<T>,
    +Sub<T>,
    +Neg<T>,
>(
    score: T, prob_a: T, prob_b: T
) -> T {
    let val = score * prob_a + prob_b;

    let mut v = NumberTrait::one()
        / (NumberTrait::one() + NumberTrait::exp(-NumberTrait::abs(val)));

    v = if val < NumberTrait::zero() {
        NumberTrait::one() - v
    } else {
        v
    };

    NumberTrait::one() - v
}

fn max(a: usize, b: usize) -> usize {
    if a > b {
        return a;
    };

    b
}

fn min<T, +Copy<T>, +Drop<T>, +PartialOrd<T>,>(a: Span<T>) -> T {
    let mut min = *a.at(0);

    let mut i = 0;
    while i != a.len() {
        if min > *a.at(i) {
            min = *a.at(i);
        }

        i += 1;
    };

    min
}

fn dot_start_end<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +Add<T>, +TensorTrait<T>, +AddEq<T>, +Mul<T>,
>(
    pA: Span<T>, pB: Span<T>, a_start: usize, a_end: usize, b_start: usize, b_end: usize
) -> T {
    let mut sum = NumberTrait::zero();
    let mut index_a = a_start;
    let mut index_b = b_start;
    while index_a != a_end
        && index_b != b_end {
            sum = sum + *pA.at(index_a) * *pB.at(index_b);
            index_a += 1;
            index_b += 1;
        };

    sum
}

fn sv_dot<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +Add<T>, +TensorTrait<T>, +AddEq<T>, +Mul<T>,
>(
    pA: Span<T>, pB: Span<T>
) -> T {
    let mut i = 0;
    let mut sum = NumberTrait::zero();
    while i != pA.len() {
        sum = sum + *pA.at(i) * *pB.at(i);
        i += 1;
    };

    sum
}

fn squared_diff<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +Add<T>,
    +TensorTrait<T>,
    +AddEq<T>,
    +Mul<T>,
    +Sub<T>,
>(
    pA: Span<T>, pB: Span<T>
) -> T {
    let mut i = 0;
    let mut sum = NumberTrait::zero();
    while i != pA
        .len() {
            sum = sum + (*pA.at(i) - *pB.at(i)).pow(NumberTrait::one() + NumberTrait::one());
            i += 1;
        };

    sum
}

fn dot<T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +Mul<T>, +AddEq<T>, +Add<T>, +Div<T>>(
    ref self: NullableVec<T>, ref vec: NullableVec<T>
) -> T {
    assert(self.len == vec.len, 'wrong vec len for dot prod');
    let n = self.len;
    let mut sum: T = NumberTrait::zero();
    let mut i = 0_usize;
    while i != n {
        sum += self.at(i) * vec.at(i);
        i += 1;
    };

    sum
}

fn div_element_wise<T, MAG, +Mul<T>, +Add<T>, +Div<T>, +NumberTrait<T, MAG>, +Drop<T>, +Copy<T>>(
    ref self: NullableVec<T>, elem: T
) {
    let m = self.len;

    let mut i = 0_usize;
    while i != m {
        VecTrait::set(ref self, i, VecTrait::at(ref self, i) / elem);
        i += 1;
    };
}

