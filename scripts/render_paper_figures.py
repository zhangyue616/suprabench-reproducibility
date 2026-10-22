"""Render frozen numerical displays and export the Figure 1 assets.

No training, predictions, scoring or model selection are performed. Outputs
must be new paths; repository outputs are restricted to ignored runs/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
DEFAULT = ("1", "3", "4", "5", "6", "S1", "S2", "S3")
BUILDERS = {
    "3": ("canonical", "figure3"), "S3": ("canonical", "figure_s3"),
    "4": ("seed_similarity", "figure4"), "5": ("seed_similarity", "figure5"),
    "6": ("family", "build"),
    "S1": ("supplementary", "figure_s1"), "S2": ("supplementary", "figure_s2"),
}


def output_directory(argument):
    output = (Path(argument).expanduser().resolve() if argument else
              REPO / "runs" / ("figures_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")))
    if output.exists():
        raise ValueError("Output already exists; select a fresh directory")
    if output == REPO or output in REPO.parents:
        raise ValueError("Output cannot contain the repository")
    if output.is_relative_to(REPO) and not output.is_relative_to(REPO / "runs"):
        raise ValueError("Repository output is permitted only below runs/")
    if output == REPO / "runs":
        raise ValueError("Select a fresh child of runs/, not runs/ itself")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", default=",".join(DEFAULT),
                        help="Comma-separated: 1,3,4,5,6,S1,S2,S3 (default: all)")
    parser.add_argument("--formats", default="png,pdf,svg",
                        help="Comma-separated subset of png,pdf,svg; Figure 1 provides png,pdf only")
    parser.add_argument("--output", help="A new output directory; default: runs/figures_TIMESTAMP")
    parser.add_argument("--pdf-rasterizer-python",
                        help="Retained for CLI compatibility; Figure 1 assets need no rasterizer")
    args = parser.parse_args()
    selected = [v.strip().upper() for v in args.figures.split(",") if v.strip()]
    formats = [v.strip().lower() for v in args.formats.split(",") if v.strip()]
    if "2" in selected:
        parser.error("Figure 2's complete portable hybrid production chain is not recovered. "
                     "Its scientific inputs remain available; no substitute artwork is generated.")
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(DEFAULT):
        parser.error("Select each supported figure at most once: " + ",".join(DEFAULT))
    if not formats or len(set(formats)) != len(formats) or set(formats) - {"png", "pdf", "svg"}:
        parser.error("--formats must be a unique subset of png,pdf,svg")
    try:
        output = output_directory(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    from figure_rendering import common
    effective_formats = {
        figure: [suffix for suffix in formats
                 if figure != "1" or suffix in common.FIGURE1_FORMATS]
        for figure in selected
    }
    for figure, supported in effective_formats.items():
        if not supported:
            parser.error("Figure " + figure + " has no supported requested format; "
                         "Figure 1 provides PNG/PDF assets only")
    skipped_formats = {figure: [suffix for suffix in formats
                               if suffix not in effective_formats[figure]]
                       for figure in selected}
    common.configure(output, formats, args.pdf_rasterizer_python)
    # Imports and their external dependencies are resolved before creating output.
    modules = {name: importlib.import_module("figure_rendering." + name)
               for name, _ in {BUILDERS[f] for f in selected if f != "1"}}
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "RUNNING", "selected_figures": selected, "formats": formats,
               "requested_formats": formats,
               "numerical_rendering_defaults": {"font_family": common.FONT_FAMILY,
                                                "font_fallback": common.FONT_FALLBACK,
                                                "png_dpi": 600},
               "training": False, "prediction": False, "rescoring": False,
               "new_scientific_aggregation": False, "figures": {}}
    try:
        for figure in selected:
            if skipped_formats[figure]:
                print("Figure_" + figure + ": skipping unsupported format(s): " +
                      ", ".join(skipped_formats[figure]), file=sys.stderr)
            if figure == "1":
                result = common.render_figure1()
            else:
                module, function = BUILDERS[figure]
                result = getattr(modules[module], function)()
            stem = "Figure_" + figure
            paths = [output / f"{stem}.{suffix}" for suffix in effective_formats[figure]]
            if not all(p.is_file() and p.stat().st_size for p in paths):
                raise RuntimeError(stem + " did not produce every effective format")
            receipt["figures"][figure] = {
                "requested_formats": formats,
                "effective_formats": effective_formats[figure],
                "skipped_formats": skipped_formats[figure],
                "outputs": [{"file": p.name, "bytes": p.stat().st_size} for p in paths],
                "display_checks": result,
            }
            print(stem + ": " + ", ".join(p.name for p in paths))
        receipt["status"] = "COMPLETED"
    except Exception as exc:
        receipt["status"], receipt["error_type"] = "FAILED", type(exc).__name__
        raise
    finally:
        receipt["versions"] = {}
        for name in ("matplotlib", "numpy", "pandas", "Pillow", "svglib", "reportlab"):
            try:
                receipt["versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                receipt["versions"][name] = "unavailable"
        (output / "RENDERING_RECEIPT.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Output: " + str(output))


if __name__ == "__main__":
    main()
