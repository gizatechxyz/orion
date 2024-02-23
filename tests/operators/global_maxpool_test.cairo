use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use core::debug::PrintTrait;
use orion::operators::tensor::math::global_maxpool::global_maxpool;
#[test]
#[available_gas(200000000000)]
fn global_maxpool_test() {
    let data = array![
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96
    ];
    let a = TensorTrait::<u32>::new(array![2, 3, 4, 4].span(), data.span());
    let actual = global_maxpool(@a);

    // shape test
    assert(*actual.data.at(0) == 2, 'N must be 2');
    assert(*actual.data.at(1) == 3, 'C must be 3');
    assert(*actual.data.at(2) == 1, 'H must be 1');
    assert(*actual.data.at(3) == 1, 'W must be 1');

    let expected = TensorTrait::<
        u32
    >::new(array![2, 3, 1, 1].span(), array![16, 32, 48, 64, 80, 96].span());

    let eq = actual.equal(@expected);

    assert(*eq.data.at(0) == 1);
    assert(*eq.data.at(1) == 1);
    assert(*eq.data.at(2) == 1);
    assert(*eq.data.at(3) == 1);
    assert(*eq.data.at(4) == 1);
    assert(*eq.data.at(5) == 1);
}
