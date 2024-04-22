use core::array::ArrayTrait;
use core::clone::Clone;
use core::traits::Into;
use core::array::SpanTrait;
use core::dict::Felt252DictTrait;
use core::dict::Felt252DictEntryTrait;
use orion::numbers::FP16x16;

use orion::operators::tensor::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;
use orion::operators::tensor::{I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FP32x32, FP32x32Impl, FixedTrait};

use core::debug::PrintTrait;
use orion::operators::nn::{NNTrait, FP16x16NN};
use orion::operators::ml::POST_TRANSFORM;

#[derive(Destruct)]
struct LinearRegressor<T> {
    coefficients: Span<T>,
    intercepts: Option<Span<T>>,
    target: usize,
    post_transform: POST_TRANSFORM,
}


/// Trait
///
/// predict - Performs the generalized linear regression evaluation.
trait LinearRegressorTrait<T> {
    /// # LinearRegressorTrait::predict
    ///
    /// ```rust 
    ///    fn predict(regressor: LinearRegressor<T>, X: Tensor<T>) -> Tensor<T>;
    /// ```
    ///
    /// Linear Regressor. Performs the generalized linear regression evaluation.
    /// 
    /// ## Args
    ///
    /// * `regressor`: LinearRegressor<T> - A LinearRegressor object.
    /// * `X`:  Input 2D tensor.
    ///
    /// ## Returns
    ///
    /// * Tensor<T> containing the generalized linear regression evaluation of the input X.
    ///
    /// ## Type Constraints
    ///
    /// `LinearRegressor` and `X` must be fixed points
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor, FP16x16TensorAdd};
    /// use orion::operators::ml::linear::linear_regressor::{
    ///     LinearRegressorTrait, POST_TRANSFORM, LinearRegressor
    /// };
    /// use orion::numbers::{FP16x16, FixedTrait};
    /// use orion::operators::nn::{NNTrait, FP16x16NN};
    /// 
    /// fn example_linear_regressor() -> Tensor<FP16x16> {
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
    ///     let coefficients: Span<FP16x16> = array![
    ///         FP16x16 { mag: 19661, sign: false },
    ///         FP16x16 { mag: 50463, sign: true },
    /// 
    ///     ]
    ///         .span();
    /// 
    ///     let intercepts: Span<FP16x16> = array![
    ///         FP16x16 { mag: 32768, sign: false },
    /// 
    ///     ]
    ///         .span();
    ///     let intercepts = Option::Some(intercepts);    
    /// 
    ///     let target : usize = 1;
    ///     let post_transform = POST_TRANSFORM::NONE;
    /// 
    ///     let mut regressor: LinearRegressor<FP16x16> = LinearRegressor {
    ///         coefficients,
    ///         intercepts,
    ///         target,
    ///         post_transform
    ///     };
    /// 
    ///     let scores = LinearRegressorTrait::predict(regressor, X);
    /// 
    ///     scores
    /// }
    /// 
    /// >>> 
    /// [[-0.27], [-1.21], [-2.15]]
    /// 
    /// 
    /// 
    /// fn example_linear_regressor_2() -> Tensor<FP16x16> {
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
    ///     let coefficients: Span<FP16x16> = array![
    ///         FP16x16 { mag: 19661, sign: false },
    ///         FP16x16 { mag: 50463, sign: true },
    ///         FP16x16 { mag: 19661, sign: false },
    ///         FP16x16 { mag: 50463, sign: true },
    /// 
    ///     ]
    ///         .span();
    /// 
    ///     let intercepts: Span<FP16x16> = array![
    ///         FP16x16 { mag: 32768, sign: false },
    ///         FP16x16 { mag: 45875, sign: false },
    /// 
    ///     ]
    ///         .span();
    ///     let intercepts = Option::Some(intercepts);  
    /// 
    ///     let target = 2;
    ///     let post_transform = POST_TRANSFORM::NONE;
    /// 
    ///     let mut regressor: LinearRegressor<FP16x16> = LinearRegressor {
    ///         coefficients,
    ///         intercepts,
    ///         target,
    ///         post_transform
    ///     };
    /// 
    ///     let scores = LinearRegressorTrait::predict(regressor, X);
    /// 
    ///     scores
    /// }
    /// 
    /// >>>
    /// [[-0.27, -0.07], [-1.21, -1.01], [-2.15, -1.95]]   
    /// ```
    ///
    ///

    fn predict(regressor: LinearRegressor<T>, X: Tensor<T>) -> Tensor<T>;
}

impl LinearRegressorImpl<
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
    +PrintTrait<T>,
    +AddEq<T>,
    +Div<T>,
    +Mul<T>,
    +Add<Tensor<T>>,
    +NNTrait<T>,
> of LinearRegressorTrait<T> {
    fn predict(regressor: LinearRegressor<T>, X: Tensor<T>) -> Tensor<T> {
        let n: usize = regressor.coefficients.len() / regressor.target;
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(regressor.target);
        shape.append(n);
        let mut coefficients = TensorTrait::new(shape.span(), regressor.coefficients);

        let coefficients = coefficients.transpose(array![1, 0].span());
        let mut score = X.matmul(@coefficients);

        match regressor.intercepts {
            Option::Some(intercepts) => {
                let mut shape: Array<usize> = array![];
                shape.append(1);
                shape.append(intercepts.len());
                let intercepts = TensorTrait::new(shape.span(), intercepts);
                score = TensorTrait::add(score, intercepts);
            },
            Option::None => {},
        };

        // Post Transform
        let score = match regressor.post_transform {
            POST_TRANSFORM::NONE => score, // No action required
            POST_TRANSFORM::SOFTMAX => NNTrait::softmax(@score, Option::Some(1)),
            POST_TRANSFORM::LOGISTIC => NNTrait::sigmoid(@score),
            POST_TRANSFORM::SOFTMAXZERO => NNTrait::softmax_zero(@score, 1),
            POST_TRANSFORM::PROBIT => core::panic_with_felt252('Probit not supported yet'),
        };

        score
    }
}
