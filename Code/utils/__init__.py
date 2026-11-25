"""Shared utilities; optional imaging/training dependencies load on first use.

Keep the historical package exports while allowing audit preflight to run before
NumPy, TIFF readers, or PyTorch are installed or imported.
"""
from importlib import import_module

_EXPORTS = {
    "image_utils": ("normalize_to_uint8", "load_image_as_pil_rgb", "load_tif_as_pil_rgb"),
    "data_utils": ("seed_everything", "infer_label", "resolve_paths_inplace", "CELL_FEATURES",
                   "add_cell_aggregates", "MultiModalTiles", "PREFERRED_STATS_COLS"),
    "benchmark_utils": ("add_explicit_label_columns", "add_qc_columns", "build_prediction_views",
                        "compute_binary_metrics", "ensure_identity_columns", "infer_image_id",
                        "infer_strain_id", "is_qc_pass", "json_ready", "summarize_prediction_views"),
}
__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name):
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f".{module}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
