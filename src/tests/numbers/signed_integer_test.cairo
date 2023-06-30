// use core::traits::Into;
// use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};

// #[test]
// fn test_into() {
//     let x = IntegerTrait::<i32>::new(42, false);
//     assert(x.into() == 42, 'should convert to felt252');
//     let x = IntegerTrait::<i32>::new(42, true);
//     assert(x.into() == -42, 'should convert to felt252');
// }

// #[test]
// fn test_add() {
//     // Test addition of two positive integers
//     let a = IntegerTrait::<i32>::new(42, false);
//     let b = IntegerTrait::<i32>::new(13, false);
//     let result = a + b;
//     assert(result.mag == 55, '42 + 13 = 55');
//     assert(result.sign == false, '42 + 13 -> positive');

//     // Test addition of two negative integers
//     let a = IntegerTrait::<i32>::new(42, true);
//     let b = IntegerTrait::<i32>::new(13, true);
//     let result = a + b;
//     assert(result.mag == 55, '-42 - 13 = -55');
//     assert(result.sign == true, '-42 - 13 -> negative');

//     // Test addition of a positive integer and a negative integer with the same magnitude

//     let a = IntegerTrait::<i32>::new(42, false);
//     let b = IntegerTrait::<i32>::new(42, true);
//     let result = a + b;
//     assert(result.mag == 0, '42 - 42 = 0');
//     assert(result.sign == false, '42 - 42 -> positive');

//     // Test addition of a positive integer and a negative integer with different magnitudes
//     let a = IntegerTrait::<i32>::new(42, false);
//     let b = IntegerTrait::<i32>::new(13, true);
//     let result = a + b;
//     assert(result.mag == 29, '42 - 13 = 29');
//     assert(result.sign == false, '42 - 13 -> positive');

//     // Test addition of a negative integer and a positive integer with different magnitudes
//     let a = IntegerTrait::<i32>::new(42, true);
//     let b = IntegerTrait::<i32>::new(13, false);
//     let result = a + b;
//     assert(result.mag == 29, '-42 + 13 = -29');
//     assert(result.sign == true, '-42 + 13 -> negative');
// }

// #[test]
// fn test_sub() {
//     // Test subtraction of two positive integers with larger first
//     let a = IntegerTrait::<i32>::new(42, false);
//     let b = IntegerTrait::<i32>::new(13, false);
//     let result = a - b;
//     assert(result.mag == 29, '42 - 13 = 29');
//     assert(result.sign == false, '42 - 13 -> positive');

//     // Test subtraction of two positive integers with larger second
//     let a = IntegerTrait::<i32>::new(13, false);
//     let b = IntegerTrait::<i32>::new(42, false);
//     let result = a - b;
//     assert(result.mag == 29, '13 - 42 = -29');
//     assert(result.sign == true, '13 - 42 -> negative');

//     // Test subtraction of two negative integers with larger first
//     let a = IntegerTrait::<i32>::new(42, true);
//     let b = IntegerTrait::<i32>::new(13, true);
//     let result = a - b;
//     assert(result.mag == 29, '-42 - -13 = 29');
//     assert(result.sign == true, '-42 - -13 -> negative');

//     // Test subtraction of two negative integers with larger second
//     let a = IntegerTrait::<i32>::new(13, true);
//     let b = IntegerTrait::<i32>::new(42, true);
//     let result = a - b;
//     assert(result.mag == 29, '-13 - -42 = 29');
//     assert(result.sign == false, '-13 - -42 -> positive');

//     // Test subtraction of a positive integer and a negative integer with the same magnitude
//     let a = IntegerTrait::<i32>::new(42, false);
//     let b = IntegerTrait::<i32>::new(42, true);
//     let result = a - b;
//     assert(result.mag == 84, '42 - -42 = 84');
//     assert(result.sign == false, '42 - -42 -> postive');

//     // Test subtraction of a negative integer and a positive integer with the same magnitude
//     let a = IntegerTrait::<i32>::new(42, true);
//     let b = IntegerTrait::<i32>::new(42, false);
//     let result = a - b;
//     assert(result.mag == 84, '-42 - 42 = -84');
//     assert(result.sign == true, '-42 - 42 -> negative');

//     // Test subtraction of a positive integer and a negative integer with different magnitudes
//     let a = IntegerTrait::<i32>::new(100, false);
//     let b = IntegerTrait::<i32>::new(42, true);
//     let result = a - b;
//     assert(result.mag == 142, '100 - - 42 = 142');
//     assert(result.sign == false, '100 - - 42 -> postive');

//     // Test subtraction of a negative integer and a positive integer with different magnitudes
//     let a = IntegerTrait::<i32>::new(42, true);
//     let b = IntegerTrait::<i32>::new(100, false);
//     let result = a - b;
//     assert(result.mag == 142, '-42 - 100 = -142');
//     assert(result.sign == true, '-42 - 100 -> negative');

//     // Test subtraction resulting in zero
//     let a = IntegerTrait::<i32>::new(42, false);
//     let b = IntegerTrait::<i32>::new(42, false);
//     let result = a - b;
//     assert(result.mag == 0, '42 - 42 = 0');
//     assert(result.sign == false, '42 - 42 -> positive');
// }

// #[test]
// fn test_mul() {
//     // Test multiplication of positive integers
//     let a = IntegerTrait::<i32>::new(10, false);
//     let b = IntegerTrait::<i32>::new(5, false);
//     let result = a * b;
//     assert(result.mag == 50, '10 * 5 = 50');
//     assert(result.sign == false, '10 * 5 -> positive');

//     // Test multiplication of negative integers
//     let a = IntegerTrait::<i32>::new(10, true);
//     let b = IntegerTrait::<i32>::new(5, true);
//     let result = a * b;
//     assert(result.mag == 50, '-10 * -5 = 50');
//     assert(result.sign == false, '-10 * -5 -> positive');

//     // Test multiplication of positive and negative integers
//     let a = IntegerTrait::<i32>::new(10, false);
//     let b = IntegerTrait::<i32>::new(5, true);
//     let result = a * b;
//     assert(result.mag == 50, '10 * -5 = -50');
//     assert(result.sign == true, '10 * -5 -> negative');

//     // Test multiplication by zero
//     let a = IntegerTrait::<i32>::new(10, false);
//     let b = IntegerTrait::<i32>::new(0, false);
//     let result = a * b;
//     assert(result.mag == 0, '10 * 0 = 0');
//     assert(result.sign == false, '10 * 0 -> positive');
// }

// #[test]
// fn test_div_no_rem() {
//     // Test division of positive integers
//     let a = IntegerTrait::<i32>::new(10, false);
//     let b = IntegerTrait::<i32>::new(5, false);
//     let result = a / b;
//     assert(result.mag == 2, '10 // 5 = 2');
//     assert(result.sign == false, '10 // 5 -> positive');

//     // Test division of negative integers
//     let a = IntegerTrait::<i32>::new(10, true);
//     let b = IntegerTrait::<i32>::new(5, true);
//     let result = a / b;
//     assert(result.mag == 2, '-10 // -5 = 2');
//     assert(result.sign == false, '-10 // -5 -> positive');

//     // Test division of positive and negative integers
//     let a = IntegerTrait::<i32>::new(10, false);
//     let b = IntegerTrait::<i32>::new(5, true);
//     let result = a / b;
//     assert(result.mag == 2, '10 // -5 = -2');
//     assert(result.sign == true, '10 // -5 -> negative');

//     // Test division with a = zero
//     let a = IntegerTrait::<i32>::new(0, false);
//     let b = IntegerTrait::<i32>::new(10, false);
//     let result = a / b;
//     assert(result.mag == 0, '0 // 10 = 0');
//     assert(result.sign == false, '0 // 10 -> positive');

//     // Test division with a = zero
//     let a = IntegerTrait::<i32>::new(0, false);
//     let b = IntegerTrait::<i32>::new(10, false);
//     let result = a / b;
//     assert(result.mag == 0, '0 // 10 = 0');
//     assert(result.sign == false, '0 // 10 -> positive');
// }

// #[test]
// #[available_gas(20000000)]
// fn test_div_rem() {
//     // Test division and remainder of positive integers
//     let a = IntegerTrait::<i32>::new(13, false);
//     let b = IntegerTrait::<i32>::new(5, false);
//     let (q, r) = a.div_rem(b);
//     assert(q.mag == 2 && r.mag == 3, '13 // 5 = 2 r 3');
//     assert((q.sign == false) & (r.sign == false), '13 // 5 -> positive');

//     // Test division and remainder of negative integers
//     let a = IntegerTrait::<i32>::new(13, true);
//     let b = IntegerTrait::<i32>::new(5, true);
//     let (q, r) = a.div_rem(b);
//     assert(q.mag == 2 && r.mag == 3, '-13 // -5 = 2 r -3');
//     assert(q.sign == false && r.sign == true, '-13 // -5 -> positive');

//     // Test division and remainder of positive and negative integers
//     let a = IntegerTrait::<i32>::new(13, false);
//     let b = IntegerTrait::<i32>::new(5, true);
//     let (q, r) = a.div_rem(b);
//     assert(q.mag == 3 && r.mag == 2, '13 // -5 = -3 r -2');
//     assert(q.sign == true && r.sign == true, '13 // -5 -> negative');

//     // Test division with a = zero
//     let a = IntegerTrait::<i32>::new(0, false);
//     let b = IntegerTrait::<i32>::new(10, false);
//     let (q, r) = a.div_rem(b);
//     assert(q.mag == 0 && r.mag == 0, '0 // 10 = 0 r 0');
//     assert(q.sign == false && r.sign == false, '0 // 10 -> positive');

//     // Test division and remainder with a negative dividend and positive divisor
//     let a = IntegerTrait::<i32>::new(13, true);
//     let b = IntegerTrait::<i32>::new(5, false);
//     let (q, r) = a.div_rem(b);
//     assert(q.mag == 3 && r.mag == 2, '-13 // 5 = -3 r 2');
//     assert(q.sign == true && r.sign == false, '-13 // 5 -> negative');
// }


