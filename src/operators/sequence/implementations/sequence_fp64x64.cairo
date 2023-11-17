use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::fixed_point::implementations::fp64x64::core::FP64x64;
use orion::operators::tensor::implementations::tensor_fp64x64::FP64x64Tensor;


impl FP64x64Sequence of SequenceTrait<FP64x64> {
    fn sequence_construct(tensors: Array<Tensor<FP64x64>>) -> Array<Tensor<FP64x64>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<FP64x64>> {
        functional::sequence_empty::sequence_empty::<FP64x64>()
    }
}
