"""Feeder topology helpers that work from the standard data format only.

From the line list (from, to, r, x) we derive, for every bus:
  depth           number of lines between the bus and the substation
  path_r, path_x  total resistance / reactance of the path to the substation (ohm)
  n_downstream    number of buses fed through this bus
  branch          which lateral the bus is on (the first bus after the main line split)
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def bus_features(line_from, line_to, line_r, line_x, root: str = "bus_0") -> pd.DataFrame:
    children = defaultdict(list)
    for f, t, r, x in zip(line_from, line_to, line_r, line_x):
        children[str(f)].append((str(t), float(r), float(x)))

    rows = {root: {"depth": 0, "path_r": 0.0, "path_x": 0.0, "parent": None}}
    order = [root]
    q = deque([root])
    while q:
        b = q.popleft()
        for c, r, x in children[b]:
            rows[c] = {"depth": rows[b]["depth"] + 1, "path_r": rows[b]["path_r"] + r,
                       "path_x": rows[b]["path_x"] + x, "parent": b}
            order.append(c)
            q.append(c)

    down = {b: 0 for b in order}
    for b in reversed(order):
        p = rows[b]["parent"]
        if p is not None:
            down[p] += down[b] + 1

    df = pd.DataFrame.from_dict(rows, orient="index")
    df["n_downstream"] = pd.Series(down)
    return df.loc[order]


def path_matrix(line_from, line_to, bus_names, root: str = "bus_0") -> np.ndarray:
    """A[i, k] = 1 if line k lies on the path from the substation to bus_names[i]."""
    parent_line = {str(t): k for k, t in enumerate(line_to)}
    parent_bus = {str(t): str(f) for f, t in zip(line_from, line_to)}
    A = np.zeros((len(bus_names), len(line_to)))
    for i, b in enumerate(bus_names):
        b = str(b)
        while b != root and b in parent_line:
            A[i, parent_line[b]] = 1.0
            b = parent_bus[b]
    return A
