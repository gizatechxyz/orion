use alexandria_data_structures::vec::{VecTrait, NullableVec, NullableVecImpl};

impl VecCopy<T> of Copy<NullableVec<T>>;
impl NullCopy<T> of Copy<Felt252Dict<Nullable<T>>>;
impl VecDrop<T> of Drop<NullableVec<T>>;
impl NullDrop<T> of Drop<Felt252Dict<Nullable<T>>>;

#[derive(Copy, Drop)]
struct MutMatrix<T> {
    data: NullableVec<T>,
    rows: usize,
    cols: usize,
}

#[generate_trait]
impl MutMatrixImpl<T, +Drop<T>, +Copy<T>> of MutMatrixTrait<T> {
    // Constructor for the Matrix
    fn new(rows: usize, cols: usize) -> MutMatrix<T> {
        MutMatrix { data: NullableVecImpl::new(), rows: rows, cols: cols }
    }

    // Get the value at (row, col)
    fn get(ref self: MutMatrix<T>, row: usize, col: usize) -> Option<T> {
        if row >= self.rows || col >= self.cols {
            Option::None
        } else {
            self.data.get(row * self.cols + col)
        }
    }

    // Set the value at (row, col)
    fn set(ref self: MutMatrix<T>, row: usize, col: usize, value: T) {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;
            self.data.set(index, value)
        }
    }

    // Returns the shape of the matrix as (rows, cols)
    fn shape(self: MutMatrix<T>) -> (usize, usize) {
        (self.rows, self.cols)
    }
}
