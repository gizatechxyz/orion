mod core;
mod implementations;
mod functional;

use orion::operators::nn::core::NNTrait;

use orion::operators::nn::implementations::nn_fp8x23::FP8x23NN;
use orion::operators::nn::implementations::nn_fp16x16::FP16x16NN;
use orion::operators::nn::implementations::nn_fp32x32::FP32x32NN;
use orion::operators::nn::implementations::nn_fp64x64::FP64x64NN;
use orion::operators::nn::implementations::nn_i8::I8NN;
use orion::operators::nn::implementations::nn_i32::I32NN;
use orion::operators::nn::implementations::nn_u32::U32NN;
