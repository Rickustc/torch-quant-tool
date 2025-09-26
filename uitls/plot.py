"""Utility for comparing floating-point and QDQ-quantized tensor distributions."""
from __future__ import annotations
import argparse
import os
from typing import Iterable, Tuple, Union
import numpy as np
from bokeh import plotting
from bokeh.models import Band, ColumnDataSource, Span
TensorLike = Union[np.ndarray, "torch.Tensor", Iterable[float]]
def _tensor_to_numpy(tensor: TensorLike) -> np.ndarray:
    """Convert supported tensor-like inputs to a NumPy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    # Import torch lazily so the utility works even when PyTorch is not installed.
    try:
        import torch
    except ImportError:  # pragma: no cover - torch not available in some environments
        torch = None
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)
def _compute_pdf(tensor: np.ndarray, bins: int, range_: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return histogram bin centers and the corresponding probability density."""
    flattened = tensor.reshape(-1)
    counts, edges = np.histogram(flattened, bins=bins, range=range_, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts
def compare_fp_and_qdq_tensors(
    float_tensor: TensorLike,
    qdq_tensor: TensorLike,
    results_dir: str,
    title: str = "tensor_histogram",
    num_bins: int = 1024,
) -> plotting.figure:
    """Export an overlaid histogram for a floating-point tensor and its QDQ version.
    Args:
        float_tensor: Tensor captured before quantization.
        qdq_tensor: Tensor after applying QDQ quantization and dequantization.
        results_dir: Directory where the HTML file will be written.
        title: Title of the histogram as well as the filename stem.
        num_bins: Number of histogram bins to use for both tensors.
    Returns:
        The generated Bokeh figure instance.
    """
    os.makedirs(results_dir, exist_ok=True)
    fp_np = _tensor_to_numpy(float_tensor).astype(np.float64)
    qdq_np = _tensor_to_numpy(qdq_tensor).astype(np.float64)
    combined_min = float(np.minimum(fp_np.min(), qdq_np.min()))
    combined_max = float(np.maximum(fp_np.max(), qdq_np.max()))
    centers_fp, pdf_fp = _compute_pdf(fp_np, num_bins, (combined_min, combined_max))
    centers_qdq, pdf_qdq = _compute_pdf(qdq_np, num_bins, (combined_min, combined_max))
    filename = os.path.join(results_dir, f"{title}.html")
    plotting.output_file(filename)
    figure = plotting.figure(height=500, title=title)
    source_fp = ColumnDataSource(data={"entries": centers_fp, "pdfs": pdf_fp})
    figure.line("entries", "pdfs", source=source_fp, color="navy", legend_label="FP32 PDF")
    figure.add_layout(
        Band(base="entries", upper="pdfs", source=source_fp, level="underlay", fill_color="navy", fill_alpha=0.3)
    )
    source_qdq = ColumnDataSource(data={"entries": centers_qdq, "pdfs": pdf_qdq})
    figure.line("entries", "pdfs", source=source_qdq, color="orange", legend_label="QDQ PDF")
    figure.add_layout(
        Band(
            base="entries",
            upper="pdfs",
            source=source_qdq,
            level="underlay",
            fill_color="orange",
            fill_alpha=0.3,
        )
    )
    for value, color, label in (
        (fp_np.min(), "green", "FP32 MIN"),
        (fp_np.max(), "green", "FP32 MAX"),
        (qdq_np.min(), "red", "QDQ MIN"),
        (qdq_np.max(), "red", "QDQ MAX"),
    ):
        span = Span(location=float(value), dimension="height", line_color=color, line_dash="dashed")
        figure.add_layout(span)
        figure.line([], [], line_dash="dashed", line_color=color, legend_label=label)
    figure.legend.click_policy = "hide"
    plotting.save(figure)
    return figure
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare floating-point and QDQ tensor histograms.")
    parser.add_argument("float_tensor", type=str, help="Path to a .npy file containing the floating-point tensor.")
    parser.add_argument("qdq_tensor", type=str, help="Path to a .npy file containing the QDQ tensor.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory where the histogram HTML will be saved.")
    parser.add_argument("--title", type=str, default="tensor_histogram", help="Plot title and filename stem.")
    parser.add_argument("--bins", type=int, default=1024, help="Number of histogram bins to use.")
    return parser.parse_args()
def main() -> None:
    args = _parse_args()
    float_tensor = np.load(args.float_tensor)
    qdq_tensor = np.load(args.qdq_tensor)
    compare_fp_and_qdq_tensors(float_tensor, qdq_tensor, args.output_dir, args.title, args.bins)
if __name__ == "__main__":
    weight = torch.randn(1,256,14,14)
    weight_qdq = torch.quantize_per_tensor(weight, scale=0.25)
    