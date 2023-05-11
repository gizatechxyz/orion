use array::ArrayTrait;

// Fake macro to compute gas left
// TODO: Remove when automatically handled by compiler.
#[inline(always)]
fn check_gas() {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(_) => {},
        Option::None(_) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        }
    }
}

fn u32_max(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}
