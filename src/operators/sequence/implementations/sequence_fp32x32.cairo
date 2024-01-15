use core::option::OptionTrait;

use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::fixed_point::implementations::fp32x32::core::FP32x32;
use orion::operators::tensor::implementations::tensor_fp32x32::FP32x32Tensor;
use orion::operators::tensor::implementations::tensor_i32::I32Tensor;


impl FP32x32Sequence of SequenceTrait<FP32x32> {
    fn sequence_construct(tensors: Array<Tensor<FP32x32>>) -> Array<Tensor<FP32x32>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<FP32x32>> {
        functional::sequence_empty::sequence_empty::<FP32x32>()
    }

    fn sequence_length(self: Array<Tensor<FP32x32>>) -> Tensor<u32> {
        functional::sequence_length::sequence_length(self)
    }

    fn sequence_at(sequence: Array<Tensor<FP32x32>>, position: Tensor<i32>) -> Tensor<FP32x32> {
        functional::sequence_at::sequence_at(sequence, position)
    }

    fn sequence_erase(
        sequence: Array<Tensor<FP32x32>>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<FP32x32>> {
        functional::sequence_erase::sequence_erase(sequence, position)
    }

    fn sequence_insert(
        self: Array<Tensor<FP32x32>>, tensor: @Tensor<FP32x32>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<FP32x32>> {
        functional::sequence_insert::sequence_insert(self, tensor, position)
    }

    fn concat_from_sequence(
        sequence: Array<Tensor<FP32x32>>, axis: i32, new_axis: Option<usize>
    ) -> Tensor<FP32x32> {
        functional::concat_from_sequence::concat_from_sequence(sequence, axis, new_axis)
    }
}
