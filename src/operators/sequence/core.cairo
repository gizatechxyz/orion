use orion::operators::tensor::core::Tensor;

/// Trait
///
/// sequence_construct - Constructs a tensor sequence containing the input tensors.
/// sequence_empty - Returns an empty tensor sequence.
trait SequenceTrait<T> {
    /// ## sequence.sequence_construct
    ///
    /// ```rust 
    ///    fn sequence_construct(tensors: Array<Tensor<T>>) -> Array<Tensor<T>>;
    /// ```
    ///
    /// Constructs a tensor sequence containing the input tensors.
    ///
    /// ## Args
    ///
    /// * `tensors`(`Array<Tensor<T>>`) - The array of input tensors.
    ///
    /// ## Panics 
    /// 
    /// * Panics if input tensor array is empty.
    ///
    /// ## Returns
    ///
    /// A tensor sequence `Array<Tensor<T>>` containing the input tensors.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// use orion::operators::sequence::SequenceTrait;
    ///
    /// fn sequence_construct_example() -> Array<Tensor<usize>> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());
    ///     let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![4, 5, 6, 7].span());
    ///     let result = SequenceTrait::sequence_construct(tensors: array![tensor1, tensor2]);
    ///     return result;
    /// }
    /// >>> [[0, 1, 2, 3], [4, 5, 6, 7]]
    /// ```
    ///
    fn sequence_construct(tensors: Array<Tensor<T>>) -> Array<Tensor<T>>;
    /// ## sequence.sequence_empty
    ///
    /// ```rust
    ///    fn sequence_empty() -> Array<Tensor<T>>;
    /// ```
    ///
    /// Returns an empty tensor sequence.
    ///
    /// ## Args
    ///
    /// ## Returns
    ///
    /// An empty `Array<Tensor<T>>` instance.
    ///
    /// ## Examples
    ///
    /// Let's create a new empty sequence.
    ///
    /// ```rust
    /// use array::{ArrayTrait, SpanTrait};
    ///
    /// use orion::operators::tensor::{
    ///     TensorTrait, // we import the trait
    ///     Tensor, // we import the type
    ///     U32Tensor // we import the implementation. 
    /// };
    /// use orion::operators::sequence::SequenceTrait;
    ///
    /// fn sequence_empty_example() -> Array<Tensor<u32>> {
    ///     let sequence = SequenceTrait::sequence_empty();
    ///
    ///     return sequence;
    /// }
    ///
    /// >>> []
    /// ```
    ///
    fn sequence_empty() -> Array<Tensor<T>>;
}
