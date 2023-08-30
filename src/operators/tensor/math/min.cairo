use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::NumberTrait;

/// Cf: TensorTrait::min docstring
fn min_in_tensor<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut vec: Span::<T>
) -> T {
    let mut min_value: T = NumberTrait::max_value();

    loop {
        match vec.pop_front() {
            Option::Some(item) => {
                let check_min = min_value.min(*item);
                if (min_value > check_min) {
                    min_value = check_min;
                }
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return min_value;
}
