mod core;
mod implementations;
mod functional;

use orion::operators::nn::core::NNTrait;

use orion::operators::nn::implementations::nn_fp8x23::NN_fp8x23;
use orion::operators::nn::implementations::nn_fp16x16::NN_fp16x16;

use orion::operators::nn::implementations::nn_i8_fp8x23::NN_i8_fp8x23;
use orion::operators::nn::implementations::nn_i8_fp16x16::NN_i8_fp16x16;

use orion::operators::nn::implementations::nn_i32_fp8x23::NN_i32_fp8x23;
use orion::operators::nn::implementations::nn_i32_fp16x16::NN_i32_fp16x16;

use orion::operators::nn::implementations::nn_u32_fp8x23::NN_u32_fp8x23;
use orion::operators::nn::implementations::nn_u32_fp16x16::NN_u32_fp16x16;
