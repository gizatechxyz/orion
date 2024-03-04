use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::operators::tensor::implementations::tensor_u32::U32Tensor;
use orion::operators::tensor::implementations::tensor_i32::I32Tensor;

impl U32Sequence of SequenceTrait<u32> {
    fn sequence_construct(tensors: Array<Tensor<u32>>) -> Array<Tensor<u32>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<u32>> {
        functional::sequence_empty::sequence_empty::<u32>()
    }

    fn sequence_length(self: Array<Tensor<u32>>) -> Tensor<u32> {
        functional::sequence_length::sequence_length(self)
    }

    fn sequence_at(sequence: Array<Tensor<u32>>, position: Tensor<i32>) -> Tensor<u32> {
        functional::sequence_at::sequence_at(sequence, position)
    }

    fn sequence_erase(
        sequence: Array<Tensor<u32>>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<u32>> {
        functional::sequence_erase::sequence_erase(sequence, position)
    }

    fn sequence_insert(
        self: Array<Tensor<u32>>, tensor: @Tensor<u32>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<u32>> {
        functional::sequence_insert::sequence_insert(self, tensor, position)
    }

    fn concat_from_sequence(
        sequence: Array<Tensor<u32>>, axis: i32, new_axis: Option<usize>
    ) -> Tensor<u32> {
        functional::concat_from_sequence::concat_from_sequence(sequence, axis, new_axis)
    }
}
