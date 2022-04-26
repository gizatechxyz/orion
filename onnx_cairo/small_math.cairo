# %builtins range_check

from starkware.cairo.common.uint256 import Uint256
from starkware.cairo.common.math_cmp import is_le, is_not_zero
from starkware.cairo.common.pow import pow
from starkware.cairo.common.math import (
    assert_le,
    assert_lt,
    sqrt,
    sign,
    abs_value,
    signed_div_rem,
    unsigned_div_rem,
    assert_not_zero
)

const Small_Math_INT_PART = 2 ** 64
const Small_Math_FRACT_PART = 2 ** 61
const Small_Math_BOUND = 2 ** 125
const Small_Math_ONE = 1 * Small_Math_FRACT_PART
const Small_Math_E = 6267931151224907085

const Small_Math_PI = 7244019458077122842
const Small_Math_sqrt_PI = 4087000321264375119

# Wrapper for Fix64x61 - encouraging types
struct Fix64x61:
    member val: felt
end

# Double struct: double = mantisa / 10^pow
struct Double:
    member mantisa : felt
    member pow : felt
end

func Small_Math_assert64x61 {range_check_ptr} (x: felt):
    assert_le(x, Small_Math_BOUND)
    assert_le(-Small_Math_BOUND, x)
    return ()
end

########################################################################
############## COUPLE OF SIMPLE DISPLAYING FUNCS #######################
########################################################################

func show_Double (x: Double):
    tempvar mantisa = x.mantisa
    tempvar power = x.pow
    %{
        doub = ids.mantisa / 10 ** ids.power
        print(' Double: ' + str(doub))
    %}
    return()
end

func show_Fix (x: Fix64x61):
    tempvar mantisa = x.val
    %{
        fix = ids.mantisa / ids.Small_Math_FRACT_PART
        print(' Fix: ' + str(fix))
    %}
    return()
end

########################################################################
######################## MANAGING TYPES ################################
########################################################################

# Converts a fixed point value to a felt, truncating the fractional component
func Fix_to_Felt {range_check_ptr} (x: Fix64x61) -> (res: felt):
    let (res, _) = signed_div_rem(x.val, Small_Math_FRACT_PART, Small_Math_BOUND)
    return (res)
end

# Converts a felt to a fixed point value ensuring it will not overflow
func Felt_to_Fix {range_check_ptr} (x: felt) -> (res: Fix64x61):
    assert_le(x, Small_Math_INT_PART)
    assert_le(-Small_Math_INT_PART, x)
    tempvar x_fix = x * Small_Math_FRACT_PART
    let res:Fix64x61 = Fix64x61(x_fix)
    return (res = res)
end

# Linear transformation from base 10 to base 2 - convenient for bringing double floats from python
func Double_to_Fix {range_check_ptr}(x: Double) -> (res: Fix64x61):
    alloc_locals
    local fix_mant = x.mantisa * Small_Math_FRACT_PART
    let x_pow_divisor : felt = pow(10, x.pow)
    let (x_fix, _) = unsigned_div_rem(fix_mant, x_pow_divisor)
    return (res = Fix64x61(x_fix))
end

# TODO
# func Fix_to_Double {range_check_ptr}(x: Fix64x61) -> (res: Double):
#     return (res = Double(x_fix))
# end

# # Converts a fixed point 64.61 value to a uint256 value
# func Small_Math_toUint256 (x: felt) -> (res: Uint256):
#     let res = Uint256(low = x, high = 0)
#     return (res)
# end

# # Converts a uint256 value into a fixed point 64.61 value ensuring it will not overflow
# func Small_Math_fromUint256 {range_check_ptr} (x: Uint256) -> (res: felt):
#     assert x.high = 0
#     let (res) = Felt_to_Fix(x.low)
#     return (res)
# end

# Calculates the floor of a 64.61 value
func Small_Math_floor {range_check_ptr} (x: felt) -> (res: felt):
    let (int_val, mod_val) = signed_div_rem(x, Small_Math_ONE, Small_Math_BOUND)
    let res = x - mod_val
    Small_Math_assert64x61(res)
    return (res)
end

# Calculates the ceiling of a 64.61 value
func Small_Math_ceil {range_check_ptr} (x: felt) -> (res: felt):
    let (int_val, mod_val) = signed_div_rem(x, Small_Math_ONE, Small_Math_BOUND)
    let res = (int_val + 1) * Small_Math_ONE
    Small_Math_assert64x61(res)
    return (res)
end

# Returns the minimum of two values
func Small_Math_min {range_check_ptr} (x: felt, y: felt) -> (res: felt):
    let (x_le) = is_le(x, y)

    if x_le == 1:
        return (x)
    else:
        return (y)
    end
end

# Returns the maximum of two values
func Small_Math_max {range_check_ptr} (x: felt, y: felt) -> (res: felt):
    let (x_le) = is_le(x, y)

    if x_le == 1:
        return (y)
    else:
        return (x)
    end
end

# Convenience addition method to assert no overflow before returning
func Small_Math_add {range_check_ptr} (x: Fix64x61, y: Fix64x61) -> (res: Fix64x61):
    let res = x.val + y.val
    Small_Math_assert64x61(res)
    return (res = Fix64x61(res))
end

# Convenience subtraction method to assert no overflow before returning
func Small_Math_sub {range_check_ptr} (x: Fix64x61, y: Fix64x61) -> (res: Fix64x61):
    let res = x.val - y.val
    Small_Math_assert64x61(res)
    return (Fix64x61(res))
end

# Multiples two fixed point values and checks for overflow before returning
func Small_Math_mul {range_check_ptr} (x: Fix64x61, y: Fix64x61) -> (res: Fix64x61):
    tempvar product = x.val * y.val
    let (res, _) = signed_div_rem(product, Small_Math_FRACT_PART, Small_Math_BOUND)
    Small_Math_assert64x61(res)
    return (Fix64x61 (res))
end

# Divides two fixed point values and checks for overflow before returning
# Both values may be signed (i.e. also allows for division by negative b)
func Small_Math_div {range_check_ptr} (x: Fix64x61, y: Fix64x61) -> (res: Fix64x61):
    alloc_locals
    let (div) = abs_value(y.val)
    let (div_sign) = sign(y.val)
    tempvar product = x.val * Small_Math_FRACT_PART
    let (res_u, _) = signed_div_rem(product, div, Small_Math_BOUND)
    Small_Math_assert64x61(res_u)
    tempvar res = res_u * div_sign
    return (res = Fix64x61(res))
end

# ORIGINAL COMMENTS:
# Calclates the value of x^y and checks for overflow before returning
# x is a 64x61 fixed point value
# y is a standard felt (int)
func Small_Math__pow_int {range_check_ptr} (x: Fix64x61, y: felt) -> (res: Fix64x61):
    alloc_locals
    let (exp_sign) = sign(y)
    let (exp_val) = abs_value(y)

    if exp_sign == 0:
        return (Fix64x61(Small_Math_ONE))
    end

    if exp_sign == -1:
        let (num) = Small_Math__pow_int(Fix64x61(x.val), exp_val)
        let (res) = Small_Math_div(Fix64x61(Small_Math_ONE), num)
        return (res)
    end

    let (half_exp, rem) = unsigned_div_rem(exp_val, 2)
    let (half_pow) = Small_Math__pow_int(Fix64x61(x.val), half_exp)
    
    let (res_p) = Small_Math_mul(half_pow, half_pow)

    if rem == 0:
        Small_Math_assert64x61(res_p.val)
        return (res_p)
    else:
        let (res) = Small_Math_mul(res_p, x)
        Small_Math_assert64x61(res.val)
        return (res)
    end
end

# Calclates the value of x^y and checks for overflow before returning
# x is a 64x61 fixed point value
# y is a 64x61 fixed point value
func Small_Math_pow {range_check_ptr} (x: Fix64x61, y: Fix64x61) -> (res: Fix64x61):
    alloc_locals
    let (y_int, y_frac) = signed_div_rem(y.val, Small_Math_ONE, Small_Math_BOUND)

    # use the more performant integer pow when y is an int
    if y_frac == 0:
        return Small_Math__pow_int(x, y_int)
    end

    # x^y = exp(y*ln(x)) for x > 0 (will error for x < 0
    let (ln_x) = Small_Math_ln(x)
    let (y_ln_x) = Small_Math_mul(y,ln_x)
    let (res) = Small_Math_exp(y_ln_x)
    return (res)
    # Small_Math_assert64x61(res)
    # return (res)
end

# Calculates the square root of a fixed point value
# x must be positive
func Small_Math_sqrt {range_check_ptr} (x: felt) -> (res: felt):
    alloc_locals
    let (root) = sqrt(x)
    let (scale_root) = sqrt(Small_Math_FRACT_PART)
    let (res, _) = signed_div_rem(root * Small_Math_FRACT_PART, scale_root, Small_Math_BOUND)
    Small_Math_assert64x61(res)
    return (res)
end

# Calculates the most significant bit where x is a fixed point value
# TODO: use binary search to improve performance
func Small_Math__msb {range_check_ptr} (x: Fix64x61) -> (res: felt):
    alloc_locals

    let (cmp) = is_le(x.val, Small_Math_FRACT_PART)

    if cmp == 1:
        return (0)
    end

    let (div, _) = unsigned_div_rem(x.val, 2)
    let (rest) = Small_Math__msb(Fix64x61(div))
    local res = 1 + rest
    Small_Math_assert64x61(res)
    return (res)
end

# Calculates the binary exponent of x: 2^x
func Small_Math_exp2 {range_check_ptr} (x: Fix64x61) -> (res: Fix64x61):
    alloc_locals

    let (exp_sign) = sign(x.val)

    if exp_sign == 0:
        return (Fix64x61 (Small_Math_ONE))
    end

    let (exp_value) = abs_value(x.val)
    let (int_part, frac_part) = unsigned_div_rem(exp_value, Small_Math_FRACT_PART)
    tempvar fix_val_2 = 2 * Small_Math_ONE
    let (int_res) = Small_Math__pow_int(Fix64x61(fix_val_2), int_part)

    # 1.069e-7 maximum error
    const a1 = 2305842762765193127
    const a2 = 1598306039479152907
    const a3 = 553724477747739017
    const a4 = 128818789015678071
    const a5 = 20620759886412153
    const a6 = 4372943086487302

    let (r6) = Small_Math_mul(Fix64x61(a6), Fix64x61(frac_part))
    tempvar temp_r6_a5 = r6.val + a5
    let (r5) = Small_Math_mul(Fix64x61(temp_r6_a5), Fix64x61(frac_part))
    tempvar temp_r5_a4 = r5.val + a4
    let (r4) = Small_Math_mul(Fix64x61(temp_r5_a4), Fix64x61(frac_part))
    tempvar temp_r4_a3 = r4.val + a3
    let (r3) = Small_Math_mul(Fix64x61(temp_r4_a3), Fix64x61(frac_part))
    tempvar temp_r3_a2 = r3.val + a2
    let (r2) = Small_Math_mul(Fix64x61(temp_r3_a2), Fix64x61(frac_part))
    tempvar frac_res = r2.val + a1

    let (res_u) = Small_Math_mul(int_res, Fix64x61(frac_res))
    
    if exp_sign == -1:
        let (res_i) = Small_Math_div(Fix64x61(Small_Math_ONE), res_u)
        Small_Math_assert64x61(res_i.val)
        return (res_i)
    else:
        Small_Math_assert64x61(res_u.val)
        return (res_u)
    end
end

# Calculates the natural exponent of x: e^x
func Small_Math_exp {range_check_ptr} (x: Fix64x61) -> (res: Fix64x61):
    const mod = 3326628274461080623
    let (bin_exp) = Small_Math_mul(x, Fix64x61(mod))
    let (res) = Small_Math_exp2(bin_exp)
    return (res)
end

# Calculates the binary logarithm of x: log2(x)
# x must be greather than zero
func Small_Math_log2 {range_check_ptr} (x: Fix64x61) -> (res: Fix64x61):
    alloc_locals

    if x.val == Small_Math_ONE:
        return (Fix64x61(0))
    end

    let (is_frac) = is_le(x.val, Small_Math_FRACT_PART - 1)

    # Compute negative inverse binary log if 0 < x < 1
    if is_frac == 1:
        let (div) = Small_Math_div(Fix64x61(Small_Math_ONE), x)
        let (res_i) = Small_Math_log2(div)
        return (Fix64x61(-res_i.val))
    end

    let (x_over_two, _) = unsigned_div_rem(x.val, 2)
    let (b) = Small_Math__msb(Fix64x61(x_over_two))
    let (divisor) = pow(2, b)
    let (norm, _) = unsigned_div_rem(x.val, divisor)

    # 4.233e-8 maximum error
    const a1 = -7898418853509069178
    const a2 = 18803698872658890801
    const a3 = -23074885139408336243
    const a4 = 21412023763986120774
    const a5 = -13866034373723777071
    const a6 = 6084599848616517800
    const a7 = -1725595270316167421
    const a8 = 285568853383421422
    const a9 = -20957604075893688

    let (r9) = Small_Math_mul(Fix64x61(a9), Fix64x61(norm))
    tempvar temp_r9_a8 = r9.val + a8
    let (r8) = Small_Math_mul(Fix64x61(temp_r9_a8), Fix64x61(norm))
    tempvar temp_r8_a7 = r8.val + a7
    let (r7) = Small_Math_mul(Fix64x61(temp_r8_a7), Fix64x61(norm))
    tempvar temp_r7_a6 = r7.val + a6
    let (r6) = Small_Math_mul(Fix64x61(temp_r7_a6), Fix64x61(norm))
    tempvar temp_r6_a5 = r6.val + a5
    let (r5) = Small_Math_mul(Fix64x61(temp_r6_a5), Fix64x61(norm))
    tempvar temp_r5_a4 = r5.val + a4
    let (r4) = Small_Math_mul(Fix64x61(temp_r5_a4), Fix64x61(norm))
    tempvar temp_r4_a3 = r4.val + a3
    let (r3) = Small_Math_mul(Fix64x61(temp_r4_a3), Fix64x61(norm))
    tempvar temp_r3_a2 = r3.val + a2
    let (r2) = Small_Math_mul(Fix64x61(temp_r3_a2), Fix64x61(norm))
    local norm_res = r2.val + a1

    let (int_part: Fix64x61) = Felt_to_Fix(b)
    local res = int_part.val + norm_res
    Small_Math_assert64x61(res)
    return (Fix64x61(res))
end

# Calculates the natural logarithm of x: ln(x)
# x must be greater than zero
func Small_Math_ln {range_check_ptr} (x: Fix64x61) -> (res: Fix64x61):
    const ln_2 = 1598288580650331957
    let (log2_x) = Small_Math_log2(x)
    let (product) = Small_Math_mul(log2_x, Fix64x61(ln_2))
    tempvar prod = product.val
    
    return (product)
end

# Calculates the base 10 log of x: log10(x)
# x must be greater than zero
func Small_Math_log10 {range_check_ptr} (x: Fix64x61) -> (res: Fix64x61):
    const log10_2 = 694127911065419642
    let (log10_x) = Small_Math_log2(x)
    let (product) = Small_Math_mul(log10_x, Fix64x61(log10_2))
    return (product)
end

# Calculates hyperbolic sine of x (fixed point)
func Small_Math_sinh {range_check_ptr} (x: Fix64x61) -> (res: Fix64x61):
    alloc_locals

    let (ex) = Small_Math_exp(x)
    let fix_ONE = Fix64x61(Small_Math_ONE)
    let (ex_i) = Small_Math_div(fix_ONE, ex)
    let sub_1 : Fix64x61 = Fix64x61(ex.val - ex_i.val)
    let (res) = Small_Math_div(sub_1, Fix64x61(2 * Small_Math_ONE))
    Small_Math_assert64x61(res.val)
    return (res)
end

# Gets the factorial of a fixed point integer value using simple iteration
func Small_Math_fact {range_check_ptr} (x: Fix64x61) -> (res: Fix64x61):
    if x.val == 0:
        return (res = Fix64x61(Small_Math_ONE))
    end
    let (sub_1) = Small_Math_sub(x, Fix64x61(Small_Math_ONE))
    let (partial_fact) = Small_Math_fact (sub_1)
    let (res) = Small_Math_mul(partial_fact, x)
    return (res = res)
end

# func main {range_check_ptr}():
#     let (fix_2: Fix64x61) = Felt_to_Fix(2)
#     let (res: Fix64x61) = Small_Math_log2(fix_2)
#     %{
#         print(' Log_2(2), small_math:')
#     %}
#     show_Fix(res)
#     tempvar e_felt = 6267931151224906542
#     let (res_2) = Math64x61_ln(e_felt)
#     %{
#         print(' res:', ids.res_2/ids.Math64x61_ONE)
#     %}

#     return()
# end