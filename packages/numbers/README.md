# Orion-Numbers: A Cairo Numbers Library

## Fixed Points
A fixed point library, inspired by [Cubit lib](https://github.com/influenceth/cubit).

### Supported Implementations

| Implementations | Status |
| :-------------: | :----: |
|      16x16      |   ✅    |
|      32x32      |   ⏳    |
|      64x64      |   ⏳    |


A signed 16.16-bit fixed point number is a fraction in which the numerator is a signed 32-bit integer and the denominator is 2^16. Since the denominator stays the same there is no need to store it (as in a floating point value).
