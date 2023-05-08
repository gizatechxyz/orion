trait IntegerTrait<T, U> {
    fn new(mag: U, sign: bool) -> T;
    fn div_rem(self: T, other: T) -> (T, T);
    fn abs(self: T) -> T;
    fn max(self: T, other: T) -> T;
    fn min(self: T, other: T) -> T;
}
