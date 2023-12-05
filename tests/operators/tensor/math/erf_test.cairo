use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor};
use orion::numbers::{FixedTrait, FP16x16};
use core::box::BoxTrait;
use debug::PrintTrait;
use orion::utils::assert_eq;

#[test]
#[available_gas(1000000000)]
fn test_erf_example() {
    // The erf inputs is [1.0, 0.134, 0.520, 2.0, 3.5, 5.164]
    let tensor = TensorTrait::<FP16x16>::new(
        shape: array![6].span(),
        data: array![
            FixedTrait::new(65536, false),
            FixedTrait::new(8832, false),
            FixedTrait::new(34079, false),
            FixedTrait::new(131072, false),
            FixedTrait::new(229376, false),
            FixedTrait::new(338428, false),
        ]
            .span(),
    );
    let erf_result = tensor.erf();
    
    let mut arr: Span<FP16x16> = erf_result.data;

    loop {
        match arr.pop_front() {
            Option::Some(res) => {
                let mut res = *res;
                res.mag.print();
            },
            Option::None(_) => { break; }
        };
    };
    assert(*erf_result.data.get(0).unwrap().unbox().mag == 55227, 'f1_erf it works!');
    assert(*erf_result.data.get(1).unwrap().unbox().mag == 9560, 'f2_erf it works!');
    assert(*erf_result.data.get(2).unwrap().unbox().mag == 35252, 'f3_erf it works!');
    assert(*erf_result.data.get(3).unwrap().unbox().mag == 65229, 'f4_erf it works!');
    assert(*erf_result.data.get(4).unwrap().unbox().mag == 65536, 'f5_erf it works!');
    assert(*erf_result.data.get(5).unwrap().unbox().mag == 65536, 'f6_erf it works!');
}