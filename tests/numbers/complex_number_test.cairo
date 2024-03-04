//use cubit::f128::types::fixed::FixedTrait;
use orion::numbers::complex_number::complex_trait::ComplexTrait;
use orion::numbers::complex_number::complex64::{TWO, complex64};
use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
use core::debug::PrintTrait;


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
}

#[test]
#[available_gas(2000000000)]
fn test_exp() {
    // Test exp of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false), FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i

    let _z = ComplexTrait::exp(a);

    let _z_expected: complex64 = ComplexTrait::new(
        FixedTrait::new(402848450095324460000, true), FixedTrait::new(923082101320478400000, true)
    );
}


#[test]
#[available_gas(2000000000)]
fn test_exp2() {
    // Test exp2 of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(73786976294838206464, false), FixedTrait::new(774763251095801167872, false)
    ); // 4 + 42i

    let exp2 = a.exp2();
    assert(
        exp2.real == FixedTrait::new(197471674372309809080, true), 'exp2.real = -10.70502356986'
    );
    assert(exp2.img == FixedTrait::new(219354605088992285353, true), 'exp2.img = -11.89127707');
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
}


#[test]
#[available_gas(2000000000)]
fn test_log2() {
    // Test log2 of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let log2 = a.log2();
    assert(log2.real == FixedTrait::new(34130530934667840346, false), 'log2.real = 1.85021986');
    assert(log2.img == FixedTrait::new(26154904847122126193, false), 'log2.img = 1.41787163');
}


#[test]
#[available_gas(2000000000)]
fn test_log10() {
    // Test log10 of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let log10 = a.log10();
    assert(
        log10.real == FixedTrait::new(10274314139629458970, false),
        'log10.real = 0.5569716761534184'
    );
    assert(
        log10.img == FixedTrait::new(7873411322133748801, false), 'log10.img = 0.42682189085546657'
    );
}

#[test]
#[available_gas(2000000000)]
fn test_acos() {
    // Test acos of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let acos = a.acos();
    assert(
        acos.real == FixedTrait::new(18449430688981877061, false), 'acos.real = 1.000143542473797'
    );
    assert(acos.img == FixedTrait::new(36587032881711954470, true), 'acos.img = -1.98338702991653');
}


#[test]
#[available_gas(2000000000)]
fn test_asin() {
    // Test asin of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let asin = a.asin();
    assert(asin.real == FixedTrait::new(10526647143326614308, false), 'asin.real = 0.57065278432');
    assert(asin.img == FixedTrait::new(36587032881711954470, false), 'asin.img = 1.9833870299');
}

#[test]
#[available_gas(2000000000)]
fn test_atan() {
    // Test atan of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let atan = a.atan();
    assert(atan.real == FixedTrait::new(26008453796191787243, false), 'atan.real = 1.40992104959');
    assert(atan.img == FixedTrait::new(4225645162986888119, false), 'atan.img = 0.229072682968538');
}

#[test]
#[available_gas(2000000000)]
fn test_cos() {
    // Test cos of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let cos = a.cos();
    assert(
        cos.real == FixedTrait::new(77284883172661882094, true), 'cos.real = -4.189625690968807'
    );
    assert(cos.img == FixedTrait::new(168035443352962049425, true), 'cos.img = -9.109227893755337');
}


#[test]
#[available_gas(2000000000)]
fn test_sin() {
    // Test sin of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let sin = a.sin();
    assert(
        sin.real == FixedTrait::new(168870549816927860082, false), 'sin.real = 9.15449914691143'
    );
    assert(sin.img == FixedTrait::new(76902690389051588309, true), 'sin.img = -4.168906959966565');
}


#[test]
#[available_gas(2000000000)]
fn test_tan() {
    // Test tan of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );

    let tan = a.tan();
    assert(tan.real == FixedTrait::new(69433898428143694, true), 'tan.real = -0.003764025641');
    assert(tan.img == FixedTrait::new(18506486100303669886, false), 'tan.img = 1.00323862735361');
}


#[test]
#[available_gas(2000000000)]
fn test_acosh() {
    // Test acosh of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let acosh = a.acosh();
    assert(acosh.real == FixedTrait::new(36587032878947915965, false), 'acosh.real = 1.9833870');
    assert(acosh.img == FixedTrait::new(18449360714192945790, false), 'acosh.img = 1.0001435424');
}


#[test]
#[available_gas(2000000000)]
fn test_asinh() {
    // Test asinh of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let asinh = a.asinh();
    assert(
        asinh.real == FixedTrait::new(36314960239770126586, false), 'asinh.real = 1.96863792579'
    );
    assert(asinh.img == FixedTrait::new(17794714057579789616, false), 'asinh.img = 0.96465850440');
}


#[test]
#[available_gas(2000000000)]
fn test_atanh() {
    // Test atanh of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let atanh = a.atanh();
    assert(
        atanh.real == FixedTrait::new(2710687792925618924, false), 'atanh.real = 0.146946666225'
    );
    assert(atanh.img == FixedTrait::new(24699666646262346226, false), 'atanh.img = 1.3389725222');
}


#[test]
#[available_gas(2000000000)]
fn test_cosh() {
    // Test cosh of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let cosh = a.cosh();
    assert(cosh.real == FixedTrait::new(68705646899632870392, true), 'cosh.real = -3.72454550491');
    assert(cosh.img == FixedTrait::new(9441447324287988702, false), 'cosh.img = 0.511822569987');
}


#[test]
#[available_gas(2000000000)]
fn test_sinh() {
    // Test sinh of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let sinh = a.sinh();
    assert(sinh.real == FixedTrait::new(66234138518106676624, true), 'sinh.real = -3.59056458998');
    assert(sinh.img == FixedTrait::new(9793752294470951790, false), 'sinh.img = 0.530921086');
}


#[test]
#[available_gas(2000000000)]
fn test_tanh() {
    // Test tanh of a complex number
    let a: complex64 = ComplexTrait::new(
        FixedTrait::new(36893488147419103232, false), FixedTrait::new(55340232221128654848, false)
    );
    let tanh = a.tanh();
    assert(tanh.real == FixedTrait::new(17808227710002974080, false), 'tanh.real = 0.96538587902');
    assert(tanh.img == FixedTrait::new(182334107030204896, true), 'tanh.img = 0.009884375');
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
