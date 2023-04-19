import pyscipopt as ps
import pathlib
import numpy as np
import itertools
from typing import List, IO
import os
import argparse
import math


class NoiseDosage:
    def __init__(self, name : str, m : int, n : int, alpha : List[float], d : List[int],
                 t : List[float], total_hours : int) -> None:
        assert isinstance(name, str)
        self.name : str = name

        assert isinstance(m, int)
        # number of machines
        self.m : int = m

        assert isinstance(n, int)
        # number of workers
        self.n : int = n

        assert len(alpha) == m
        assert all(isinstance(i, float) for i in alpha)
        # noise dosage units per machine
        self.alpha : List[float] = alpha

        assert len(d) == m
        assert all(isinstance(i, int) for i in d)
        # number of work cycles per machine to be executed
        self.d : List[int] = d

        assert len(t) == m
        assert all(isinstance(i, float) for i in t)
        # number of hours of operation per work cycle
        self.t : List[float] = t

        isinstance(total_hours, int)
        # total number of hours that people can work
        self.total_hours : int = total_hours

    def write(self, fh: IO[str]) -> None:
        fh.write(f"{self.m} {self.n}" + os.linesep)
        fh.write(" ".join(map("{:e}".format, self.alpha)) + os.linesep)
        fh.write(" ".join(map(str, self.d)) + os.linesep)
        fh.write(" ".join(map("{:e}".format, self.t)) + os.linesep)
        fh.write(f"{self.total_hours}" + os.linesep)


def read_noise_dosage(path: pathlib.Path) -> NoiseDosage:
    with open(path, "r") as f:
        lines = list(f)
        m, n = map(int, lines[0].strip().split())
        alpha = list(map(float, lines[1].strip().split()))
        d = list(map(int, lines[2].strip().split()))
        t = list(map(float, lines[3].strip().split()))
        total_hours = int(lines[4].strip())
        return NoiseDosage(path.stem, m, n, alpha, d, t, total_hours)


def generate_noise_dosage(inst : NoiseDosage, with_sherali_smith_symhandling : bool = False) -> ps.Model:
    """
    From Sherali and Smith (2001):
    Suppose that we have n workers, indexed by  j = 1, ..., n,
    executing tasks on m machines, indexecb y i = 1, ..., m,
    For machine i, some d_i work cycles must be executed,
    each requiring t_i hours of operation,
    and inducing a noise dosage of alpha_i units.
    Furthermore, each worker is limited to H hours of total work,
    while performing no more than u_i work cycles on machine i,
    where u_i is either some known or logically derived upper bound.
    """
    model = ps.Model(problemName=inst.name)

    x = {
        (i, j): model.addVar(name=f"x[{i},{j}]", vtype="I", lb=0.0)
        for i in range(inst.m)
        for j in range(inst.n)
    }

    z = model.addVar(name=f"z", vtype="C", lb=0.0)

    for j in range(inst.n):
        model.addCons(z >= ps.quicksum(inst.alpha[i] * x[i, j] for i in range(inst.m)))

    for i in range(inst.m):
        model.addCons(ps.quicksum(x[i, j] for j in range(inst.n)) == inst.d[i])

    for j in range(inst.n):
        model.addCons(ps.quicksum(inst.t[i] * x[i, j] for i in range(inst.m)) <= inst.total_hours)

    if with_sherali_smith_symhandling:
        # for each machine, derive upper bounds of how much one worker can work on it.
        u = {
            i: min(inst.total_hours // inst.t[i], inst.d[i])
            for i in range(inst.m)
        }
        M = max(u.values()) + 1

        for j in range(inst.n - 1):
            model.addCons(
                ps.quicksum(M**i * x[i, j] for i in range(inst.m))
                >= ps.quicksum(M**i * x[i, j + 1] for i in range(inst.m)))

    model.setObjective(z, sense="minimize")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Generate models from data instances")
    parser.add_argument("-d", "--data-dir", default="data/")
    args = parser.parse_args()
    print(f"Generating instances from {args.data_dir}")

    data_path = pathlib.Path(__file__).parent / pathlib.Path(args.data_dir)
    output_path = pathlib.Path(__file__).parent / pathlib.Path("instances/")

    assert data_path.exists(), f"{data_path} does not exist"
    output_path.mkdir(parents=True, exist_ok=True)

    path : pathlib.Path
    for path in data_path.iterdir():
        if path.suffix:
            # data files have no suffix
            continue

        # Without Sherali-Smith's symmetry handling constraints
        writepath = output_path / pathlib.Path(f"{path.stem}.mps")
        if not writepath.exists():
            inst : NoiseDosage = read_noise_dosage(path)
            model : ps.Model = generate_noise_dosage(inst)
            model.writeProblem(writepath)
            # model.optimize()

        # With Sherali-Smith's symmetry handling constraints
        writepath = output_path / pathlib.Path(f"{path.stem}_sym.mps")
        if not writepath.exists():
            inst : NoiseDosage = read_noise_dosage(path)
            model : ps.Model = generate_noise_dosage(inst, with_sherali_smith_symhandling=True)
            model.writeProblem(writepath)
            # model.optimize()
