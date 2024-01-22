use orion::operators::tensor::core::Tensor;

/// Trait
///
/// sequence_construct - Constructs a tensor sequence containing the input tensors.
/// sequence_empty - Returns an empty tensor sequence.
/// sequence_length - Returns the length of the input sequence.
/// sequence_insert - Insert a tensor into a sequence.
/// sequence_at - Outputs the tensor at the specified position in the input sequence.
/// sequence_erase – Outputs the tensor sequence with the erased tensor at the specified position.
/// concat_from_sequence - Concatenate a sequence of tensors into a single tensor.
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
    /// # tensor.sequence_length
    ///
    /// ```rust
    ///    fn sequence_length(self: Array<Tensor<T>>) -> Tensor<u32>;
    /// ```
    ///
    /// Returns the length of the input sequence.
    ///
    /// ## Args
    ///
    /// * `self`(`Array<Tensor<T>>`) - The input sequence.
    ///
    /// ## Returns
    ///
    /// The length of the sequence as scalar, i.e. a tensor of shape [].
    ///
    /// ## Examples
    ///
    /// Let's create new u32 Tensor with constant 42.
    ///
    /// ```rust
    /// let mut sequence = ArrayTrait::new();
    ///
    /// let mut shape = ArrayTrait::<usize>::new();
    /// shape.append(1);
    /// shape.append(2);
    ///
    /// let mut data = ArrayTrait::new();
    /// data.append(3);
    /// data.append(1);
    ///
    /// sequence.append(TensorTrait::new(shape.span(), data.span()));
    ///
    /// sequence.sequence_length()
    /// >>> [1]
    /// ```
    ///
    fn sequence_length(self: Array<Tensor<T>>) -> Tensor<u32>;
    /// # tensor.sequence_insert
    ///
    /// ```rust 
    ///    fn sequence_insert(self: Array<Tensor<T>>, tensor: @Tensor<T>, position: Option<Tensor<i32>>) -> Array<Tensor<T>>;
    /// ```
    ///
    /// Returns a tensor sequence that inserts 'tensor' into 'self' at 'position'.
    ///
    /// ## Args
    ///
    /// * `self`(`Array<Tensor<T>>`) - input sequence.
    /// * `tensor` (`@Tensor<T>`) - the tensor to insert.
    /// * `position` (`@Tensor<i32>`) - the index for insertion (default: -1).
    ///
    /// ## Returns
    ///
    /// Tensor sequence containing 'tensor' inserted into 'self' at 'position'.
    ///
    /// ## Examples
    ///
    /// Let's insert the tensor [2] into the sequence [[1], [3]] at position 1.
    /// use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor, U32Tensor};
    ///
    /// fn sequence_insert_example() -> Array<Tensor<u32>> {
    ///     // Prepare sequence
    ///     let mut sequence = ArrayTrait::new();
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(1);
    ///
    ///     let mut data = ArrayTrait::new();
    ///     data.append(1);
    ///     sequence.append(TensorTrait::new(shape.span(), data.span()));
    ///     let mut data = ArrayTrait::new();
    ///     data.append(3);
    ///
    ///     sequence.append(TensorTrait::new(shape.span(), data.span()));
    ///
    ///     // Prepare input tensor
    ///     let mut data = ArrayTrait::new();
    ///     data.append(2);
    ///     let tensor = TensorTrait::new(shape.span(), data.span());
    ///
    ///     // Prepare position
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     let mut data = ArrayTrait::<i32>::new();
    ///     data.append(i32 { mag: 1, sign: false });
    ///     let position = TensorTrait::<i32>::new(shape.span(), data.span())
    ///
    ///     let sequence = self.sequence_insert(tensor, Option::Some(position));
    ///
    ///     return sequence;
    /// }
    ///
    /// >>> [[1], [2], [3]]
    /// ```
    ///
    fn sequence_insert(
        self: Array<Tensor<T>>, tensor: @Tensor<T>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<T>>;
    /// ## tensor.sequence_at
    ///
    /// ```rust 
    ///    fn sequence_at(sequence: Array<Tensor<T>>, position: Tensor<i32>) -> Tensor<T>;
    /// ```
    ///
    /// Outputs the tensor at the specified position in the input sequence.
    ///
    /// ## Args
    ///
    /// * `tensors`(`Array<Tensor<T>>`) - The tensor sequence.
    /// * `position`(`Tensor<i32>`) - The position tensor.
    ///
    /// ## Panics 
    /// 
    /// * Panics if position is not a scalar
    /// * Panics if position is out of bounds [-n, n - 1]
    ///
    /// ## Returns
    ///
    /// The tensor `Tensor<T>` from the sequence at the specified position.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor, I32Tensor};
    /// use orion::numbers::{i32, IntegerTrait};
    ///
    /// fn sequence_at_example() -> Tensor<u32> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());
    ///     let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![4, 5, 6, 7].span());
    ///     
    ///     let mut sequence = ArrayTrait::new();
    ///     sequence.append(tensor1);
    ///     sequence.append(tensor2);
    ///
    ///     let position = TensorTrait::new(shape: array![].span(), data: array![IntegerTrait::new(1, false)].span());
    ///
    ///     let result = TensorTrait::sequence_at(sequence, position);
    ///     return result;
    /// }
    /// >>> [4, 5, 6, 7]
    /// ```
    ///
    fn sequence_at(sequence: Array<Tensor<T>>, position: Tensor<i32>) -> Tensor<T>;
    /// ## tensor.sequence_erase
    ///
    /// ```rust 
    ///    fn sequence_erase(sequence: Array<Tensor<T>>, position: Option<Tensor<i32>>) -> Array<Tensor<T>>;
    /// ```
    ///
    /// Outputs the tensor sequence with the erased tensor at the specified position.
    ///
    /// ## Args
    ///
    /// * `tensors`(`Array<Tensor<T>>`) - The tensor sequence.
    /// * `position`(`Option<Tensor<i32>>`) - The optional position tensor (by default erases the last tensor).
    ///
    /// ## Panics 
    /// 
    /// * Panics if position is not a scalar
    /// * Panics if position is out of bounds [-n, n - 1]
    ///
    /// ## Returns
    ///
    /// The tensor sequence `Array<Tensor<T>>` with the erased tensor at the specified position.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor, I32Tensor};
    /// use orion::numbers::{i32, IntegerTrait};
    ///
    /// fn sequence_erase_example() -> Tensor<u32> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());
    ///     let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![4, 5, 6, 7].span());
    ///     let tensor3 = TensorTrait::new(shape: array![2, 2].span(), data: array![8, 9, 10, 11].span());
    ///     
    ///     let mut sequence = ArrayTrait::new();
    ///     sequence.append(tensor1);
    ///     sequence.append(tensor2);
    ///     sequence.append(tensor3);
    ///
    ///     let position = TensorTrait::new(shape: array![].span(), data: array![IntegerTrait::new(1, false)].span());
    ///
    ///     let result = TensorTrait::sequence_erase(sequence, position);
    ///     return result;
    /// }
    /// >>> [[0, 1, 2, 3], [8, 9, 10, 11]]
    /// ```
    ///
    fn sequence_erase(
        sequence: Array<Tensor<T>>, position: Option<Tensor<i32>>
    ) -> Array<Tensor<T>>;
    /// # tensor.concat_from_sequence
    ///
    /// ```rust 
    ///    fn concat_from_sequence(sequence: Array<Tensor<T>>, axis: i32, new_axis: Option<usize>) -> Tensor<T>;
    /// ```
    ///
    /// Concatenate a sequence of tensors into a single tensor.
    ///
    /// ## Args
    ///
    /// * `sequence`(`Array<Tensor<T>>`) - The input sequence.
    /// * `axis`(`i32`) -  Axis to concat on.
    /// * `new_axis`(`Option<usize>`) -  Optionally added new axis.
    ///
    /// ## Panics
    ///
    /// * Panics if new_axis not 0 or 1 (if value provided).
    /// * Panics if axis not in accepted ranges.
    /// * Panics if sequence length is not greater than 1.
    ///
    /// ## Returns 
    ///
    /// A new `Tensor<T>` concatenated tensor from the input tensor sequence.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use core::array::{ArrayTrait, SpanTrait};
    /// 
    /// use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
    /// 
    /// fn concat_example() -> Tensor<u32> {
    ///     let tensor1 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    ///     let tensor2 = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span(),);
    ///
    ///     let mut sequence = ArrayTrait::new();
    ///     sequence.append(tensor1);
    ///     sequence.append(tensor2);
    ///
    ///     let result = TensorTrait::concat_from_sequence(sequence: sequence, axis: 0, new_axis: Option::Some(0));
    ///     return result;
    /// }
    /// >>> [[0. 1.]
    ///      [2. 3.],
    ///      [0. 1.]
    ///      [2. 3.]]
    ///
    ///     result.shape
    /// >>> (4, 2)
    ///
    ///    let result = TensorTrait::concat_from_sequence(sequence: sequence, axis: 1, new_axis: Option::Some(0));
    ///    return result;
    /// }
    /// >>> [[0. 1., 0., 1.]
    ///      [2. 3., 2., 3.]]
    ///
    ///     result.shape
    /// >>> (2, 4 ) 
    /// ```
    ///
    fn concat_from_sequence(
        sequence: Array<Tensor<T>>, axis: i32, new_axis: Option<usize>
    ) -> Tensor<T>;
}
