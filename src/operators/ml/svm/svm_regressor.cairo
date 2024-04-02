use core::traits::TryInto;
use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::traits::Into;
use orion::numbers::NumberTrait;
use orion::operators::tensor::{
    TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor, BoolTensor
};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};
use core::debug::PrintTrait;
use orion::operators::nn::{NNTrait, FP16x16NN};
use orion::utils::get_row;
use orion::operators::ml::POST_TRANSFORM;

use orion::operators::ml::svm::core::{kernel_dot, KERNEL_TYPE};

#[derive(Copy, Drop, Destruct)]
struct SVMRegressor<T> {
    coefficients: Span<T>,
    kernel_params: Span<T>,
    kernel_type: KERNEL_TYPE,
    n_supports: usize,
    one_class: usize,
    post_transform: POST_TRANSFORM,
    rho: Span<T>,
    support_vectors: Span<T>,
}

#[derive(Copy, Drop)]
enum MODE {
    SVM_LINEAR,
    SVM_SVC,
}

/// Trait
///
/// predict - Returns the regressed values for each input in N.
trait SVMRegressorTrait<T> {
    /// # SVMRegressorTrait::predict
    ///
    /// ```rust 
    ///    fn predict(ref self: SVMRegressor<T>, X: Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Support Vector Machine regression prediction and one-class SVM anomaly detection.
    /// 
    /// ## Args
    ///
    /// * `self`: SVMRegressor<T> - A SVMRegressor object.
    /// * `X`:  Input 2D tensor.
    ///
    /// ## Returns
    ///
    /// * Tensor<T> containing the Support Vector Machine regression prediction and one-class SVM anomaly detection of the input X.
    ///
    /// ## Type Constraints
    ///
    /// `SVMRegressor` and `X` must be fixed points
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::FP16x16;
    /// use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
    /// use orion::operators::tensor::FP16x16TensorPartialEq;
    /// 
    /// use orion::operators::ml::svm::svm_regressor::{SVMRegressorTrait, POST_TRANSFORM, SVMRegressor};
    /// use orion::operators::ml::svm::core::{KERNEL_TYPE};
    /// 
    /// fn example_svm_regressor_linear() -> Tensor<FP16x16> {
    ///     let coefficients: Span<FP16x16> = array![
    ///         FP16x16 { mag: 65536, sign: false },
    ///         FP16x16 { mag: 65536, sign: true },
    ///         FP16x16 { mag: 54959, sign: false },
    ///         FP16x16 { mag: 54959, sign: true },
    ///         FP16x16 { mag: 29299, sign: false },
    ///         FP16x16 { mag: 65536, sign: true },
    ///         FP16x16 { mag: 36236, sign: false }
    ///     ]
    ///         .span();
    ///     let n_supports: usize = 7;
    ///     let one_class: usize = 0;
    ///     let rho: Span<FP16x16> = array![FP16x16 { mag: 35788, sign: false }].span();
    ///     let support_vectors: Span<FP16x16> = array![
    ///         FP16x16 { mag: 8421, sign: true },
    ///         FP16x16 { mag: 5842, sign: false },
    ///         FP16x16 { mag: 4510, sign: false },
    ///         FP16x16 { mag: 5202, sign: true },
    ///         FP16x16 { mag: 14783, sign: true },
    ///         FP16x16 { mag: 17380, sign: true },
    ///         FP16x16 { mag: 60595, sign: false },
    ///         FP16x16 { mag: 1674, sign: true },
    ///         FP16x16 { mag: 38669, sign: true },
    ///         FP16x16 { mag: 63803, sign: false },
    ///         FP16x16 { mag: 87720, sign: true },
    ///         FP16x16 { mag: 22236, sign: false },
    ///         FP16x16 { mag: 61816, sign: false },
    ///         FP16x16 { mag: 34267, sign: true },
    ///         FP16x16 { mag: 36418, sign: false },
    ///         FP16x16 { mag: 27471, sign: false },
    ///         FP16x16 { mag: 28421, sign: false },
    ///         FP16x16 { mag: 69270, sign: true },
    ///         FP16x16 { mag: 152819, sign: false },
    ///         FP16x16 { mag: 4065, sign: false },
    ///         FP16x16 { mag: 62274, sign: true }
    ///     ]
    ///         .span();
    ///     let post_transform = POST_TRANSFORM::NONE;
    ///     let kernel_params: Span<FP16x16> = array![
    ///         FP16x16 { mag: 27812, sign: false },
    ///         FP16x16 { mag: 0, sign: false },
    ///         FP16x16 { mag: 196608, sign: false }
    ///     ]
    ///         .span();
    ///     let kernel_type = KERNEL_TYPE::LINEAR;
    /// 
    ///     let mut regressor: SVMRegressor<FP16x16> = SVMRegressor {
    ///         coefficients,
    ///         kernel_params,
    ///         kernel_type,
    ///         n_supports,
    ///         one_class,
    ///         post_transform,
    ///         rho,
    ///         support_vectors,
    ///     };
    /// 
    ///     let mut X: Tensor<FP16x16> = TensorTrait::new(
    ///         array![3, 3].span(),
    ///         array![
    ///             FP16x16 { mag: 32768, sign: true },
    ///             FP16x16 { mag: 26214, sign: true },
    ///             FP16x16 { mag: 19660, sign: true },
    ///             FP16x16 { mag: 13107, sign: true },
    ///             FP16x16 { mag: 6553, sign: true },
    ///             FP16x16 { mag: 0, sign: false },
    ///             FP16x16 { mag: 6553, sign: false },
    ///             FP16x16 { mag: 13107, sign: false },
    ///             FP16x16 { mag: 19660, sign: false },
    ///         ]
    ///             .span()
    ///     );
    /// 
    ///     return SVMRegressorTrait::predict(ref regressor, X);
    /// }
    /// 
    /// >>> [[-0.468206], [0.227487], [0.92318]]
    /// ```
    ///
    ///
    fn predict(ref self: SVMRegressor<T>, X: Tensor<T>) -> Tensor<T>;
}

impl SVMRegressorImpl<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Add<T>,
    +TensorTrait<T>,
    +PrintTrait<T>,
    +AddEq<T>,
    +Div<T>,
    +Mul<T>,
    +Neg<T>,
    +Sub<T>,
    +NNTrait<T>,
> of SVMRegressorTrait<T> {
    fn predict(ref self: SVMRegressor<T>, X: Tensor<T>) -> Tensor<T> {
        let (mode_, kernel_type_, sv) = if self.n_supports > 0 {
            let mode_ = MODE::SVM_SVC;
            let kernel_type_ = self.kernel_type;
            let sv = TensorTrait::new(
                array![self.n_supports, self.support_vectors.len() / self.n_supports].span(),
                self.support_vectors
            ); //self.atts.support_vectors.reshape((self.atts.n_supports, -1))
            (mode_, kernel_type_, sv)
        } else {
            let mode_ = MODE::SVM_LINEAR;
            let kernel_type_ = KERNEL_TYPE::LINEAR;
            let sv = TensorTrait::new(
                array![self.support_vectors.len()].span(), self.support_vectors
            );
            (mode_, kernel_type_, sv)
        };

        let mut z: Array<T> = array![];
        let mut n = 0;
        while n != *X
            .shape
            .at(0) {
                let mut s = NumberTrait::zero();
                match mode_ {
                    MODE::SVM_LINEAR => {
                        let mut x_n = get_row(@X, n);
                        s = kernel_dot(self.kernel_params, x_n, self.coefficients, kernel_type_);
                        s += *self.rho.at(0);
                    },
                    MODE::SVM_SVC => {
                        let mut x_n = get_row(@X, n);
                        let mut j = 0;
                        while j != self
                            .n_supports {
                                let mut sv_j = get_row(@sv, j);
                                let d = kernel_dot(self.kernel_params, x_n, sv_j, kernel_type_);
                                s += *self.coefficients.at(j) * d;
                                j += 1;
                            };

                        s += *self.rho.at(0);
                    },
                }
                if self.one_class == 1 {
                    let elem = if s > NumberTrait::zero() {
                        NumberTrait::one()
                    } else {
                        -NumberTrait::one()
                    };
                    z.append(elem);
                } else {
                    z.append(s);
                };

                n += 1;
            };

        // Post Transform
        let mut score = TensorTrait::new(array![*X.shape.at(0)].span(), z.span());

        score = match self.post_transform {
            POST_TRANSFORM::NONE => score,
            POST_TRANSFORM::SOFTMAX => NNTrait::softmax(@score, Option::Some(1)),
            POST_TRANSFORM::LOGISTIC => NNTrait::sigmoid(@score),
            POST_TRANSFORM::SOFTMAXZERO => NNTrait::softmax_zero(@score, 1),
            POST_TRANSFORM::PROBIT => core::panic_with_felt252('Probit not supported yet'),
        };

        score
    }
}

