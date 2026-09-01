from src.utils.memory_estimator import kv_cache_gib, parameter_storage_gib


def test_parameter_storage_scales_with_precision() -> None:
    fp16 = parameter_storage_gib(1_000_000_000, 16)
    int4 = parameter_storage_gib(1_000_000_000, 4)
    assert fp16 == int4 * 4


def test_kv_cache_is_positive_and_scales_with_sequence_length() -> None:
    short = kv_cache_gib(
        layers=24,
        kv_heads=2,
        head_dim=64,
        sequence_length=512,
    )
    long = kv_cache_gib(
        layers=24,
        kv_heads=2,
        head_dim=64,
        sequence_length=1024,
    )
    assert short > 0
    assert long == short * 2

