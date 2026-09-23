from collections import Counter

def summarize(rows):
    if not rows:
        raise ValueError("No predictions")
    n = len(rows)
    def rate(key):
        return sum(bool(r[key]) for r in rows) / n
    groups = {}
    for level in sorted({r["difficulty"] for r in rows}):
        subset = [r for r in rows if r["difficulty"] == level]
        correct = sum(r["execution_correct"] for r in subset)
        groups[level] = {"count": len(subset), "correct": correct, "execution_accuracy": correct / len(subset)}
    return {"sample_count": n, "format_valid_rate": rate("format_valid"),
            "execution_success_rate": rate("execution_success"),
            "execution_accuracy": rate("execution_correct"),
            "correct_count": sum(r["execution_correct"] for r in rows),
            "by_difficulty": groups,
            "mean_input_tokens": sum(r["input_tokens"] for r in rows) / n,
            "mean_output_tokens": sum(r["output_tokens"] for r in rows) / n,
            "mean_latency_seconds": sum(r["latency_seconds"] for r in rows) / n,
            "truncated_count": sum(r["truncated"] for r in rows),
            "error_counts": dict(Counter(r["error_type"] for r in rows if r["error_type"])),
            "comparison": "single SQLite snapshot; gold row order; fixed column order; unordered bag matching; abs tolerance 1e-6",
            "difficulty_source": "frozen heuristic_v1, not official Spider difficulty"}
