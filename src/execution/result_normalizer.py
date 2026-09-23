"""Gold-order comparison; unordered comparison retains duplicate multiplicity."""
from __future__ import annotations
import math

def cell_equal(a, b, tolerance):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
    return type(a) is type(b) and a == b

def results_equal(predicted, gold, *, order_sensitive, tolerance=1e-6,
                  predicted_columns=None, gold_columns=None):
    if predicted_columns is not None and gold_columns is not None and predicted_columns != gold_columns:
        return False
    if len(predicted) != len(gold):
        return False
    def row_equal(a, b):
        return len(a) == len(b) and all(cell_equal(x, y, tolerance) for x, y in zip(a, b))
    if order_sensitive:
        return all(row_equal(a, b) for a, b in zip(predicted, gold))
    # Maximum bipartite matching avoids greedy errors around overlapping tolerances.
    edges = [[j for j, b in enumerate(gold) if row_equal(a, b)] for a in predicted]
    match = {}
    for start in range(len(predicted)):
        queue, previous, visited = [start], {}, {start}
        free = None
        for left in queue:
            for right in edges[left]:
                if right in previous:
                    continue
                previous[right] = left
                if right not in match:
                    free = right
                    break
                nxt = match[right]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
            if free is not None:
                break
        if free is None:
            return False
        while free is not None:
            left = previous[free]
            old = next((r for r, l in match.items() if l == left), None)
            match[free] = left
            free = old
    return True
