#[derive(Copy, Drop)]
struct i33 {
    inner: u32,
    sign: bool,
}

fn __add(a: i33, b: i33) -> i33 {
    if a.sign == b.sign {
        // If the signs are the same, add the magnitudes and return the result with the same sign
        let sum = a.inner + b.inner;
        return i33 { inner: sum, sign: a.sign };
    } else {
        // If the signs are different, subtract the smaller magnitude from the larger magnitude
        // and return the result with the sign of the larger magnitude
        let (larger, smaller) = if a.inner >= b.inner {
            (a, b)
        } else {
            (b, a)
        };
        let difference = larger.inner - smaller.inner;

        if difference == 0_u32 {
            return i33 { inner: 0_u32, sign: false };
        } else {
            return i33 { inner: difference, sign: larger.sign };
        }
    }
}

impl i33Add of Add::<i33> {
    fn add(a: i33, b: i33) -> i33 {
        __add(a, b)
    }
}


fn __sub(a: i33, b: i33) -> i33 {
    let neg_b = i33 { inner: b.inner, sign: !b.sign };
    a + neg_b
}

impl i33Sub of Sub::<i33> {
    fn sub(a: i33, b: i33) -> i33 {
        __sub(a, b)
    }
}


fn __mul(a: i33, b: i33) -> i33 {
    if (a.inner == 0_u32 | b.inner == 0_u32) {
        return i33 { inner: 0_u32, sign: false };
    }

    let sign = a.sign ^ b.sign;
    let inner = a.inner * b.inner;
    return i33 { inner, sign };
}

impl i33Mul of Mul::<i33> {
    fn mul(a: i33, b: i33) -> i33 {
        __mul(a, b)
    }
}

fn div_no_rem(a: i33, b: i33) -> i33 {
    if (a.inner == 0_u32) {
        return i33 { inner: 0_u32, sign: false };
    }

    let sign = a.sign ^ b.sign;

    if (sign == false) {
        return i33 { inner: a.inner / b.inner, sign: sign };
    }

    //check if result is integer 
    if (a.inner % b.inner == 0_u32) {
        return i33 { inner: a.inner / b.inner, sign: sign };
    }

    let quotient = (a.inner * 10_u32) / b.inner;
    let last_digit = quotient % 10_u32;

    if (last_digit <= 5_u32) {
        return i33 { inner: quotient / 10_u32, sign: sign };
    } else {
        return i33 { inner: (quotient / 10_u32) + 1_u32, sign: sign };
    }
}

impl i33Div of Div::<i33> {
    fn div(a: i33, b: i33) -> i33 {
        div_no_rem(a, b)
    }
}

fn modulo(a: i33, b: i33) -> i33 {
    return a - (b * (a / b));
}

impl i33Rem of Rem::<i33> {
    fn rem(a: i33, b: i33) -> i33 {
        modulo(a, b)
    }
}

fn div_rem(a: i33, b: i33) -> (i33, i33) {
    let quotient = div_no_rem(a, b);
    let remainder = modulo(a, b);

    return (quotient, remainder);
}

fn __eq(a: i33, b: i33) -> bool {
    if a.sign == b.sign & a.inner == b.inner {
        return true;
    }

    return false;
}

fn __ne(a: i33, b: i33) -> bool {
    if a.sign != b.sign | a.inner != b.inner {
        return true;
    }

    return false;
}

impl i33PartialEq of PartialEq::<i33> {
    fn eq(a: i33, b: i33) -> bool {
        __eq(a, b)
    }

    fn ne(a: i33, b: i33) -> bool {
        __ne(a, b)
    }
}

fn __gt(a: i33, b: i33) -> bool {
    if (a.sign & !b.sign) {
        return false;
    }
    if (!a.sign & b.sign) {
        return true;
    }
    if (a.sign & b.sign) {
        return a.inner < b.inner;
    } else {
        return a.inner > b.inner;
    }
}

fn __lt(a: i33, b: i33) -> bool {
    if (a.sign & !b.sign) {
        return true;
    }
    if (!a.sign & b.sign) {
        return false;
    }
    if (a.sign & b.sign) {
        return a.inner > b.inner;
    } else {
        return a.inner < b.inner;
    }
}

fn __le(a: i33, b: i33) -> bool {
    if (a == b | __lt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

fn __ge(a: i33, b: i33) -> bool {
    if (a == b | __gt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

impl i33PartialOrd of PartialOrd::<i33> {
    fn le(a: i33, b: i33) -> bool {
        __le(a, b)
    }
    fn ge(a: i33, b: i33) -> bool {
        __ge(a, b)
    }

    fn lt(a: i33, b: i33) -> bool {
        __lt(a, b)
    }
    fn gt(a: i33, b: i33) -> bool {
        __gt(a, b)
    }
}

fn __neg(x: i33) -> i33 {
    return i33 { inner: x.inner, sign: !x.sign };
}

impl i33Neg of Neg::<i33> {
    fn neg(x: i33) -> i33 {
        __neg(x)
    }
}

fn abs(x: i33) -> i33 {
    return i33 { inner: x.inner, sign: false };
}

fn max(a: i33, b: i33) -> i33 {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

fn min(a: i33, b: i33) -> i33 {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}
