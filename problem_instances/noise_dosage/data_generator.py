import pyscipopt as ps
import pathlib
import numpy as np
import itertools
from typing import List
import random
import math
from statistics import NormalDist
from generate import NoiseDosage


def generate_instance(n_machines, n_workers, worker_hours, seed):
    name = f"noise{n_machines}_{n_workers}_{worker_hours}_s{seed}"

    total_hours_available = n_workers * worker_hours
    aimed_total_hours = total_hours_available / 2.

    rgen = random.Random(seed)

    # number of jobs per machine
    d = [rgen.randint(4, 10) for j in range(n_machines)]

    # Job durations.
    # We assume that it is normally distributed, but ignore negative results if they happen to pop up.
    total_jobs = sum(d)
    mean_job_duration = aimed_total_hours / total_jobs
    stdev_job_duration = mean_job_duration * .2
    dist = NormalDist(mu=mean_job_duration, sigma=stdev_job_duration)

    def get_random_job_duration():
        ret = -1.
        while ret < 0:
            ret = dist.inv_cdf(rgen.random())
        return ret

    t = [get_random_job_duration() for j in range(n_machines)]

    # alpha: units of noise
    dist = NormalDist(mu=18, sigma=4)

    def get_random_noise_dosage():
        ret = -1.
        while ret < 0:
            ret = dist.inv_cdf(rgen.random())
        return ret

    alpha = [get_random_noise_dosage() for j in range(n_machines)]

    inst = NoiseDosage(name, n_machines, n_workers, alpha, d, t, worker_hours)
    return inst

if __name__ == "__main__":
    inst : NoiseDosage

    output_path = pathlib.Path(__file__).parent / pathlib.Path("data_generated/")
    output_path.mkdir(parents=True, exist_ok=True)

    machine_worker_pairs = [(k + 3, k + 8) for k in range(9)]
    for m, n in machine_worker_pairs:
        for seed in range(5):
            inst = generate_instance(m, n, 480, seed)
            instpath = output_path / pathlib.Path(inst.name)
            with open(instpath, "w") as f:
                inst.write(f)
