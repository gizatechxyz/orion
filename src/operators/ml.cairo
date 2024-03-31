mod tree_ensemble;
mod linear;
mod svm;
mod normalizer;

use orion::operators::ml::tree_ensemble::tree_ensemble::{TreeEnsembleTrait};

use orion::operators::ml::tree_ensemble::core::{
    TreeEnsemble, TreeEnsembleAttributes, TreeEnsembleImpl, NODE_MODES
};
use orion::operators::ml::tree_ensemble::tree_ensemble_classifier::{
    TreeEnsembleClassifier, TreeEnsembleClassifierImpl, TreeEnsembleClassifierTrait
};

use orion::operators::ml::tree_ensemble::tree_ensemble_regressor::{
    TreeEnsembleRegressor, TreeEnsembleRegressorImpl, TreeEnsembleRegressorTrait, AGGREGATE_FUNCTION
};

use orion::operators::ml::linear::linear_regressor::{
    LinearRegressorTrait, LinearRegressorImpl, LinearRegressor
};

use orion::operators::ml::linear::linear_classifier::{
    LinearClassifierTrait, LinearClassifierImpl, LinearClassifier
};

use orion::operators::ml::normalizer::normalizer::{NormalizerTrait, NORM};

#[derive(Copy, Drop)]
enum POST_TRANSFORM {
    NONE,
    SOFTMAX,
    LOGISTIC,
    SOFTMAXZERO,
    PROBIT,
}

