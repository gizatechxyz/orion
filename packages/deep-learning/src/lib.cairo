pub(crate) mod ops;
pub(crate) mod utils;

pub use ops::binary::{BinaryOpMetadata, tensor_add, tensor_mul, tensor_rem};

#[derive(Drop, Copy)]
pub struct Tensor<T> {
    pub data: Span<T>,
}
