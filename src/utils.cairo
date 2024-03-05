use orion::operators::tensor::{Tensor, TensorTrait};

fn u32_max(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}

fn saturate<T, impl TCopy: Copy<T>, impl TDrop: Drop<T>, impl PartialOrdT: PartialOrd<T>>(
    min: T, max: T, x: T
) -> T {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

fn assert_eq<T, impl TPartialEq: PartialEq<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    lhs: T, rhs: T
) {
    assert(lhs == rhs, 'should be equal');
}

fn assert_seq_eq<T, impl TPartialEq: PartialEq<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    lhs: Array<T>, rhs: Array<T>
) {
    assert(lhs.len() == rhs.len(), 'should be equal');

    let mut i = 0;
    while i != lhs.len() {
        assert_eq(lhs[i], rhs[i]);
        i += 1;
    }
}

fn get_row<T, +Drop<T>, +Copy<T>>(self: @Tensor<T>, row: usize) -> Span<T> {
    assert((*self).shape.len() == 2, 'Expected a 2D tensor');

    let row_length = *self.shape[1];
    let start = row * row_length;

    (*self).data.slice(start, row_length)
}
