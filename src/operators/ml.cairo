mod tree_regressor;
mod xgboost_regressor;

use orion::operators::ml::tree_regressor::core::{TreeRegressorTrait, TreeNode};
use orion::operators::ml::tree_regressor::implementations::tree_regressor_fp16x16::FP16x16TreeRegressor;
use orion::operators::ml::tree_regressor::implementations::tree_regressor_fp8x23::FP8x23TreeRegressor;
use orion::operators::ml::tree_regressor::implementations::tree_regressor_fp32x32::FP32x32TreeRegressor;
use orion::operators::ml::tree_regressor::implementations::tree_regressor_fp64x64::FP64x64TreeRegressor;

use orion::operators::ml::xgboost_regressor::core::{XGBoostRegressorTrait};
use orion::operators::ml::xgboost_regressor::implementations::xgboost_regressor_fp16x16::FP16x16XGBoostRegressor;
use orion::operators::ml::xgboost_regressor::implementations::xgboost_regressor_fp8x23::FP8x23XGBoostRegressor;
use orion::operators::ml::xgboost_regressor::implementations::xgboost_regressor_fp32x32::FP32x32XGBoostRegressor;
use orion::operators::ml::xgboost_regressor::implementations::xgboost_regressor_fp64x64::FP64x64XGBoostRegressor;
