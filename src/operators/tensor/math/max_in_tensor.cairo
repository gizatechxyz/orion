use orion::numbers::NumberTrait;

/// Cf: TensorTrait::max_in_tensor docstring
fn max_in_tensor<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    mut vec: Span::<T>
) -> T {
    let mut max_value: T = NumberTrait::min_value();

    loop {
        match vec.pop_front() {
            Option::Some(item) => {
                let check_max = max_value.max(*item);
                if (max_value < check_max) {
                    max_value = check_max;
                }
            },
            Option::None => { break; }
        };
    };

    max_value
}
