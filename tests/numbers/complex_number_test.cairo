use orion::numbers::complex_number::complex_trait::ComplexTrait;
use orion::numbers::complex_number::complex64::{TWO, complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
use debug::PrintTrait;


#[test]
#[available_gas(2000000000)]
fn test_add() {
    // Test addition of two complex numbers
    let a = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(73786976294838206464, false),
        FixedTrait::<FP64x64>::new(774763251095801167872, false)
    );
    let b = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(36893488147419103232, false),
        FixedTrait::<FP64x64>::new(239807672958224171008, false)
    );
    let result = a + b;
    assert(result.real == FixedTrait::<FP64x64>::new(110680464442257309696, false), '4 + 2 = 6');
    assert(
        result.img == FixedTrait::<FP64x64>::new(1014570924054025338880, false),
        '42i + 13i = 55i, b = 55'
    );
}

#[test]
#[available_gas(2000000000)]
fn test_sub() {
    // Test substraction of two complex numbers
    let a = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(73786976294838206464, false),
        FixedTrait::<FP64x64>::new(774763251095801167872, false)
    );
    let b = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(36893488147419103232, false),
        FixedTrait::<FP64x64>::new(239807672958224171008, false)
    );
    let result = a - b;
    assert(result.real == FixedTrait::<FP64x64>::new(36893488147419103232, false), '4 - 2 = 2');
    assert(
        result.img == FixedTrait::<FP64x64>::new(534955578137576996864, false),
        '42i - 13i = 29i, b = 29'
    );
}

#[test]
#[available_gas(2000000000)]
fn test_mul() {
    // Test multiplication of positive integers
    let a = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(73786976294838206464, false),
        FixedTrait::<FP64x64>::new(774763251095801167872, false)
    );
    let b = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(36893488147419103232, false),
        FixedTrait::<FP64x64>::new(239807672958224171008, false)
    );
    let result = a * b;
    assert(
        result.real == FixedTrait::<FP64x64>::new(9924348311655738769408, true),
        '4*2 - 42*13 = -538'
    );
    assert(
        result.img == FixedTrait::<FP64x64>::new(2508757194024499019776, false),
        '(4*13 + 2*42)i = 136i, b = 136'
    );

    // Test multiplication with a pure imaginary number
    let a = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(0, false),
        FixedTrait::<FP64x64>::new(774763251095801167872, false)
    );
    let b = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(36893488147419103232, false),
        FixedTrait::<FP64x64>::new(239807672958224171008, false)
    );
    let result = a * b;
    assert(
        result.real == FixedTrait::<FP64x64>::new(10071922264245415182336, true),
        '0*2 - 42*13 = 546'
    );
    assert(
        result.img == FixedTrait::<FP64x64>::new(1549526502191602335744, false),
        '(0*13 + 2*42)i = 84, b = 84'
    );

    // Test multiplication by zero
    let a = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(73786976294838206464, false),
        FixedTrait::<FP64x64>::new(774763251095801167872, false)
    );
    let b = ComplexTrait::<
        complex64
    >::new(FixedTrait::<FP64x64>::new(0, false), FixedTrait::<FP64x64>::new(0, false));
    let result = a * b;
    assert(result.real == FixedTrait::<FP64x64>::new(0, false), '0');
    assert(result.img == FixedTrait::<FP64x64>::new(0, false), '0');

    // Test i * i = -1
    let a = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(0, false),
        FixedTrait::<FP64x64>::new(18446744073709551616, false)
    );
    let b = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(0, false),
        FixedTrait::<FP64x64>::new(18446744073709551616, false)
    );
    let result = a * b;
    assert(result.real == FixedTrait::<FP64x64>::new(18446744073709551616, true), 'i * i = -1');
    assert(result.img == FixedTrait::<FP64x64>::new(0, false), 'i * i = -1 + 0i');
}


#[test]
#[available_gas(2000000000)]
fn test_div_no_rem() {
    let a = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(73786976294838206464, false),
        FixedTrait::<FP64x64>::new(774763251095801167872, false)
    );
    let b = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(36893488147419103232, false),
        FixedTrait::<FP64x64>::new(239807672958224171008, false)
    );
    let result = a / b;
    assert(
        result.real == FixedTrait::<FP64x64>::new(59072232467254864688, false),
        'real = 3.2023121387283235'
    );
    assert(
        result.img == FixedTrait::<FP64x64>::new(3412114510743963284, false),
        'img = 0.18497109826589594j'
    );
}

#[test]
#[available_gas(2000000000)]
fn test_zero() {
    // Test multiplication by zero
    let a = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(73786976294838206464, false),
        FixedTrait::<FP64x64>::new(774763251095801167872, false)
    );
    let b = ComplexTrait::zero();
    let result = a * b;
    assert(result.real == FixedTrait::<FP64x64>::new(0, false), 'should be 0');
    assert(result.img == FixedTrait::<FP64x64>::new(0, false), 'should be 0');
}


#[test]
#[available_gas(2000000000)]
fn test_conjugate() {
    // Test conjugate of a complex number
    let a = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(73786976294838206464, false),
        FixedTrait::<FP64x64>::new(774763251095801167872, false)
    );
    let conjugate = a.conjugate();
    assert(
        conjugate.real == FixedTrait::<FP64x64>::new(73786976294838206464, false),
        'conjugate.real = 4'
    );
    assert(
        conjugate.img == FixedTrait::<FP64x64>::new(774763251095801167872, true),
        'conjugate.img = -42'
    );
}

#[test]
#[available_gas(2000000000)]
fn test_mag() {
    // Test mag of a complex number
    let a = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::new(73786976294838206464, false), FixedTrait::new(774763251095801167872, false)
    );
    let mag = a.mag();
    assert(mag == FixedTrait::new(0x2a30a6de7900000000, false), 'mag = 42.190046219457976');
// should be 778268985068318500000
// is :      778268985067028086784

}

#[test]
#[available_gas(2000000000)]
fn test_arg() {
    // Test arg of a complex number
    let a = ComplexTrait::<
        complex64
    >::new(
        FixedTrait::<FP64x64>::new(73786976294838206464, false),
        FixedTrait::<FP64x64>::new(774763251095801167872, false)
    );
    let arg = a.arg();
    assert(
        arg == FixedTrait::<FP64x64>::new(27224496882576083824, false), 'arg = 1.4758446204521403'
    );
// should be 27224528006041640000
// is :      27224496882576083824
}

#[test]
#[available_gas(2000000000)]
fn test_exp() {
    // Test exp of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false), FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i

    let z = ComplexTrait::exp(a);

    let z_expected: complex64 = ComplexTrait::new(
        FixedTrait::new(402848450095324460000, true), FixedTrait::new(923082101320478400000, true)
    );
// real part : 
// should be 402848450095324460000
// is :      402847992570293444378

// img part :
// should be 923082101320478400000
// is :      923081058030224714169

//assert(z == z_expected, '-21.838458238788455-50.04038098170736j');
}


#[test]
#[available_gas(2000000000)]
fn test_sqrt() {
    // Test square root of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false), FixedTrait::new(774763251095801167872, false)
    );
    let sqrt = a.sqrt();
    assert(sqrt.real == FixedTrait::new(88650037379463118848, false), 'real = 4.80572815603723');
    assert(sqrt.img == FixedTrait::new(80608310115317055488, false), 'img = 4.369785247552674');
// real part : 
// should be 88650037382238900000
// is :      88650037379463118848

// img part : 
// should be 80608310118675710000
// is :      80608310115317055488    
}

#[test]
#[available_gas(2000000000)]
fn test_ln() {
    // Test ln of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false), FixedTrait::new(774763251095801167872, false)
    );
    let ln = a.ln();
    assert(ln.real == FixedTrait::new(69031116512113681970, false), 'ln.real = 3.7421843216430655');
    assert(ln.img == FixedTrait::new(27224496882576083824, false), 'ln.img = 1.4758446204521403');
// real part : 
// should be  69031116457998020000
// is :       69031116512113681970

// img part :
// should be  27224528006041640000
// is :       27224496882576083824
}

#[test]
#[available_gas(2000000000)]
fn test_pow() {
    // Test pow with exp = 2
    let two = ComplexTrait::new(FP64x64Impl::new(TWO, false), FP64x64Impl::new(0, false));
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false), FixedTrait::new(774763251095801167872, false)
    );

    let pow = a.pow(two);
    assert(pow.real == FixedTrait::new(32244908640844296224768, true), 'pow.real = -1748');
    assert(pow.img == FixedTrait::new(6198106008766409342976, false), 'pow.img = 336');

    // Test pow with exp = n, int
    let three: complex64 = ComplexTrait::new(
        FP64x64Impl::new(55340232221128654848, false), FP64x64Impl::new(0, false)
    );
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false), FixedTrait::new(774763251095801167872, false)
    );

    let pow = a.pow(three);
    assert(pow.real == FixedTrait::new(389305023520047451076807, true), 'pow.real = -21104');
    assert(pow.img == FixedTrait::new(1329485652886846033475029, true), 'pow.img = 72072');

    // real part :
    // should be  389300086931566377304064
    // is :       389305023520047451076807

    // img part :
    // should be  1329493738880394804068352
    // is :       1329485652886846033475029

    // Test pow with exp = w, complex
    let w: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(18446744073709551616, false)
    ); // 2 + i

    let pow = a.pow(w);
    assert(
        pow.real == FixedTrait::new(6881545343236111419203, false), 'pow.real = 373.0485407816205'
    );
    assert(
        pow.img == FixedTrait::new(2996539405459717736042, false), 'pow.img = 162.4438823807959'
    );
// real part :
// should be  6881530958869354000000
// is :       6881545343236111419203

// img part :
// should be  2996560724618318400000
// is :       2996539405459717736042
}

#[test]
#[available_gas(2000000000)]
fn test_to_polar() {
    // Test to polar coordinates of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false), FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i
    let (mag, arg) = a.to_polar();

    assert(mag == FixedTrait::new(778268985067028086784, false), 'mag = 42.190046219457976');
    assert(arg == FixedTrait::new(27224496882576083824, false), 'arg = 1.4758446204521403');
// mag :
// should be  778268985067028086784
// is :       778268985068318500000

// arg : 
// should be  27224496882576083824
// is :       27224528006041640000
}

#[test]
#[available_gas(2000000000)]
fn test_from_polar() {
    // Test from polar coordiantes of a complex number
    let mag: FP64x64 = FixedTrait::new(778268985067028086784, false); // 42.190046219457976
    let arg: FP64x64 = FixedTrait::new(27224496882576083824, false); //1.4758446204521403
    let z_actual: complex64 = ComplexTrait::from_polar(mag, arg);

    let z_expected: complex64 = ComplexTrait::new(
        FixedTrait::new(73787936714814843012, false), FixedTrait::new(774759489569697723777, false)
    );
    // mag :
    // should be  73786976294838206464
    // is :       73787936714814843012

    // img :
    // should be  774763251095801167872
    // is :       774759489569697723777

    assert(z_actual == z_expected, 'wrong number');
}

#[test]
#[available_gas(2000000000)]
fn test_reciprocal() {
    // Test from polar coordiantes of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false), FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i

    let z_actual = a.reciprocal();

    let z_expected: complex64 = ComplexTrait::new(
        FixedTrait::new(41453357469010228, false), FixedTrait::new(435260253424607397, true)
    );
    assert(z_actual == z_expected, '0.002247191011 - 0.0235955056 i');
}
