import pyscipopt as ps
import pathlib
import numpy as np
import itertools
from typing import List

def is_sorted_subset(subset, superset):
    i : int = 0
    try:
        for a in subset:
            while superset[i] < a:
                i += 1
            if superset[i] == a:
                continue
            # a is not in superset.
            return False
        return True
    except IndexError as err:
        return False


def generate_covering_design(lamb : int, v: int, k: int, t: int):
    """
    Using IP formulation of Margot 2003
    Small covering designs by branch-and-cut
    """

    model = ps.Model(problemName=f"cov_{lamb}({v},{k},{t})")

    set_k = list(map(list, itertools.combinations(range(v), r=k)))
    set_t = list(map(list, itertools.combinations(range(v), r=t)))

    x = {
        j: model.addVar(name=f"x[{j}:{','.join(map(str, entries_k))}]", vtype="I", lb=0.0, ub=float(lamb))
        for j, entries_k in enumerate(set_k)
    }

    for i, entries_t in enumerate(set_t):
        model.addCons(name=f"cov[{i}:{','.join(map(str, entries_t))}]",
                      cons=ps.quicksum(x[j] for j, entries_k in enumerate(set_k)
                                       if is_sorted_subset(entries_t, entries_k)) >= lamb)

    model.setObjective(ps.quicksum(x.values()), sense="minimize")
    return model

if __name__ == "__main__":
    output_path = pathlib.Path(__file__).parent / pathlib.Path("instances/")

    output_path.mkdir(parents=True, exist_ok=True)

    lambs = list(range(2, 4))
    vs = ks = ts = list(range(1, 13))

    for lamb, v, k, t in itertools.product(lambs, vs, ks, ts):
        writepath = output_path / pathlib.Path(f"cov_{lamb}({v},{k},{t}).cip")
        if writepath.exists():
            # skip already-generated instances
            continue

        # must have v >= k >= t >= 0
        # and cases v = k or k = t are trivial
        if v <= k:
            continue
        if k <= t:
            continue

        model : ps.Model = generate_covering_design(lamb, v, k, t)
        model.writeProblem(writepath)
        # model.optimize()
