use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::fixed_point::implementations::fp32x32::core::FP32x32;
use orion::operators::tensor::implementations::tensor_fp32x32::FP32x32Tensor;


impl FP32x32Sequence of SequenceTrait<FP32x32> {
    fn sequence_construct(tensors: Array<Tensor<FP32x32>>) -> Array<Tensor<FP32x32>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<FP32x32>> {
        functional::sequence_empty::sequence_empty::<FP32x32>()
    }
}
