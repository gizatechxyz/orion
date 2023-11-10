import os
from pathlib import Path


PATH = "./tests/nodes"


class File:
    def __init__(self, path: str, empty_buffer: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer = []

        if empty_buffer:
            return

        if os.path.isfile(path):
            with self.path.open('r') as f:
                self.buffer = f.readlines()

    def dump(self):
        with self.path.open('w') as f:
            f.writelines([f"{line}\n" for line in self.buffer])


class Mod(File):
    def __init__(self):
        super().__init__(path=f"{PATH}.cairo")

    def update(self, name: str):
        statement = f"mod {name};"
        if statement not in self.buffer:
            self.buffer.append(statement)


class Test(File):
    def __init__(self, file: str):
        super().__init__(os.path.join(PATH, file), empty_buffer=True)

    @classmethod
    def template(cls, name: str, arg_cnt: int, refs: list[str], func_sig: str) -> list[str]:
        return [
            *[f"mod input_{i};" for i in range(arg_cnt)],
            *[ "mod output_0;"],
            *[ ""],
            *[ ""],
            *[f"use {ref};" for ref in refs],
            *[ ""],
            *[ "#[test]"],
            *[ "#[available_gas(2000000000)]"],
            *[f"fn test_{name}()"+" {"],
            *[f"    let input_{i} = input_{i}::input_{i}();" for i in range(arg_cnt)],
            *[ "    let z = output_0::output_0();"],
            *[ ""],
            *[f"    let y = {func_sig};"],
            *[ ""],
            *[ "    assert_eq(y, z);"],
            *[ "}"],
        ]


class Data(File):
    def __init__(self, file: str):
        super().__init__(os.path.join(PATH, file), empty_buffer=True)

    @classmethod
    def template(cls, func: str, dtype: str, refs: list[str], data: list[str], shape: tuple) -> list[str]:
        return [
            *[f"use {ref};" for ref in refs],
            *[ ""],
            *[f"fn {func}() -> Tensor<{dtype}>"+" {"],
            *[ "    let mut shape = ArrayTrait::<usize>::new();"],
            *[f"    shape.append({s});" for s in shape],
            *[ ""],
            *[ "    let mut data = ArrayTrait::new();"],
            *[f"    data.append({d});" for d in data],
            *[ "    TensorTrait::new(shape.span(), data.span())"],
            *[ "}"],
        ]

    @classmethod
    def template_sequence(cls, func: str, dtype: str, refs: list[str], data: list[list[str]], shape: list[tuple]) -> list[str]:
        def rep(s: list[tuple], d: list[list[str]]) -> list[str]:
            x = []
            for i in range(len(s)):
                x += [
                    *[ "    let mut shape = ArrayTrait::<usize>::new();"],
                    *[f"    shape.append({s});" for s in s[i]],
                    *[ ""],
                    *[ "    let mut data = ArrayTrait::new();"],
                    *[f"    data.append({d});" for d in d[i]],
                    *[ ""],
                    *[ "    sequence.append(TensorTrait::new(shape.span(), data.span()));"],
                    *[ ""],
                ]
            return x

        return [
            *[f"use {ref};" for ref in refs],
            *[ ""],
            *[f"fn {func}() -> Array<Tensor<{dtype}>>"+" {"],
            *[ "    let mut sequence = ArrayTrait::new();"],
            *[ ""],
            *rep(shape, data),
            *[ "    sequence"],
            *[ "}"],
        ]
