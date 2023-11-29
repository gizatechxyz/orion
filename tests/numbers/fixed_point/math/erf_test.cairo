use super::function::{erf};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    ONE, FP16x16, FixedTrait, FP16x16Impl, FP16x16PartialOrd, FP16x16PartialEq
};
use debug::PrintTrait;

#[test]
#[available_gas(1000000000)]
fn it_works() {
    let f1: FP16x16 = FP16x16 { mag: 65536, sign: false };
    let f2: FP16x16 = FP16x16 { mag: 8832, sign: false };
    let f1_erf: FP16x16 = erf(f1);
    let f2_erf: FP16x16 = erf(f2);
    // f1_erf.mag.print();
    // f2_erf.mag.print();
    assert(f1_erf.mag == 55227, 'f1_erf it works!');
    assert(f2_erf.mag == 9560, 'f2_erf it works!');

}