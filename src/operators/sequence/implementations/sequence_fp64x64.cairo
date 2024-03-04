use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::numbers::fixed_point::implementations::fp64x64::core::FP64x64;
use orion::operators::tensor::implementations::tensor_fp64x64::FP64x64Tensor;
use orion::operators::tensor::implementations::tensor_i32::I32Tensor;

impl FP64x64Sequence of SequenceTrait<FP64x64> {
    fn sequence_construct(tensors: Array<Tensor<FP64x64>>) -> Array<Tensor<FP64x64>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<FP64x64>> {
        functional::sequence_empty::sequence_empty::<FP64x64>()
    }

    fn sequence_length(self: Array<Tensor<FP64x64>>) -> Tensor<u32> {
        functional::sequence_length::sequence_length(self)
    }

    fn sequence_at(sequence: Array<Tensor<FP64x64>>, position: Tensor<i32>) -> Tensor<FP64x64> {
        functional::sequence_at::sequence_at(sequence, position)
    }

    fn sequence_erase(
        sequence: Array<Tensor<FP64x64>>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<FP64x64>> {
        functional::sequence_erase::sequence_erase(sequence, position)
    }

    fn sequence_insert(
        self: Array<Tensor<FP64x64>>, tensor: @Tensor<FP64x64>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<FP64x64>> {
        functional::sequence_insert::sequence_insert(self, tensor, position)
    }

    fn concat_from_sequence(
        sequence: Array<Tensor<FP64x64>>, axis: i32, new_axis: Option<usize>
    ) -> Tensor<FP64x64> {
        functional::concat_from_sequence::concat_from_sequence(sequence, axis, new_axis)
    }
}
