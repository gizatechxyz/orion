use orion::operators::tensor::core::Tensor;
use orion::operators::sequence::core::SequenceTrait;
use orion::operators::sequence::functional;
use orion::operators::tensor::implementations::tensor_bool::BoolTensor;
use orion::operators::tensor::implementations::tensor_i32::I32Tensor;

impl BoolSequence of SequenceTrait<bool> {
    fn sequence_construct(tensors: Array<Tensor<bool>>) -> Array<Tensor<bool>> {
        functional::sequence_construct::sequence_construct(tensors)
    }

    fn sequence_empty() -> Array<Tensor<bool>> {
        functional::sequence_empty::sequence_empty::<bool>()
    }

    fn sequence_length(self: Array<Tensor<bool>>) -> Tensor<u32> {
        functional::sequence_length::sequence_length(self)
    }

    fn sequence_at(sequence: Array<Tensor<bool>>, position: Tensor<i32>) -> Tensor<bool> {
        functional::sequence_at::sequence_at(sequence, position)
    }

    fn sequence_erase(
        sequence: Array<Tensor<bool>>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<bool>> {
        functional::sequence_erase::sequence_erase(sequence, position)
    }

    fn sequence_insert(
        self: Array<Tensor<bool>>, tensor: @Tensor<bool>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<bool>> {
        functional::sequence_insert::sequence_insert(self, tensor, position)
    }

    fn concat_from_sequence(
        sequence: Array<Tensor<bool>>, axis: i32, new_axis: Option<usize>
    ) -> Tensor<bool> {
        functional::concat_from_sequence::concat_from_sequence(sequence, axis, new_axis)
    }
}
