"""Behaviours the roster's agents are built from.

* ``pure``   -- a function of its input: the same arguments always return
  the same result (cacheable unless it changes something).
* ``world``  -- reads a world that moves: the result changes every
  ``period`` seconds even for the same arguments.
* ``fresh``  -- a one-off result (a new id, a created record).
* ``noisy_result`` -- usually ``pure``, but ``1 - rate`` of the time
  something the trace doesn't show (an A/B rollout) changes it.
"""
from __future__ import annotations

import math
import random

from ..model import RunCtx, stable


def ref(r: random.Random) -> str:
    return f"{r.getrandbits(40):010x}"


def pure(name: str, args: dict, **fields) -> dict:
    r = stable(name, sorted(args.items()))
    return {"ref": ref(r), **{k: (v(r) if callable(v) else v) for k, v in fields.items()}}


def world(ctx: RunCtx, name: str, args: dict, period: int, **fields) -> dict:
    bucket = int(ctx.now.timestamp()) // period
    r = stable(name, sorted(args.items()), bucket)
    return {"ref": ref(r), "as_of": bucket * period, **{k: (v(r) if callable(v) else v) for k, v in fields.items()}}


def fresh(ctx: RunCtx, **fields) -> dict:
    return {"ref": ref(ctx.rng), **{k: (v(ctx.rng) if callable(v) else v) for k, v in fields.items()}}


def noisy_result(ctx: RunCtx, rate: float, name: str, args: dict, **fields) -> dict:
    if ctx.rng.random() < rate:
        return pure(name, args, **fields)
    return pure(name, {**args, "_variant": "b"}, **fields)


def zipf(rng: random.Random, n: int, s: float = 1.1) -> int:
    """An index in ``range(n)``, low indexes far more often (repeat customers)."""
    weights = [1 / (k + 1) ** s for k in range(n)]
    return rng.choices(range(n), weights)[0]


def pool_id(rng: random.Random, prefix: str, n: int, s: float = 1.1) -> str:
    return f"{prefix}-{zipf(rng, n, s):04d}"


def uid(ctx: RunCtx, prefix: str) -> str:
    return f"{prefix}-{ctx.rng.getrandbits(32):08x}"


def band(value: float, edges: list[float], labels: list[str]) -> str:
    return next((label for edge, label in zip(edges, labels) if value < edge), labels[-1])


def chance(r: random.Random, p: float) -> bool:
    return r.random() < p


def amount(rng: random.Random, mean: float) -> float:
    return round(rng.lognormvariate(math.log(mean), 0.6), 2)
