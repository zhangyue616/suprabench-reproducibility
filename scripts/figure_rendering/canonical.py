"""Frozen display-only renderer; output handling and metadata adapted."""
from __future__ import annotations
from .common import *

MM = 1.0 / 25.4
WIDTH_MM = 165.1
INK = '#223039'
MUTED = '#51616C'
GRID = '#DDE3E7'
RF = '#226F95'
TREE = '#607887'
GINE = '#885C86'
AFP = '#A3693C'
NULL = '#4F5B64'
WHITE = '#FFFFFF'
CONNECTOR = '#B8C3C9'
GNN_BAND = '#F7F2F6'
NULL_BAND = '#F3F5F6'
RC = {'font.family': FONT_FAMILY, 'font.sans-serif': [FONT_FAMILY], 'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8.2, 'ytick.labelsize': 8.2, 'legend.fontsize': 8.0, 'axes.linewidth': 0.65, 'lines.linewidth': 0.9, 'patch.linewidth': 0.6, 'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': MUTED, 'xtick.color': INK, 'ytick.color': INK, 'axes.unicode_minus': True, 'figure.facecolor': WHITE, 'axes.facecolor': WHITE, 'savefig.facecolor': WHITE, 'savefig.edgecolor': 'none', 'svg.fonttype': 'none', 'pdf.fonttype': 42}
MODEL_ORDER = ['PAIR_RF', 'ECFP_RF', 'XGB_pair', 'XGB_ecfp', 'LGBM_pair', 'LGBM_ecfp', '3D_pair_RF', 'GINE', 'AttentiveFP']
MODEL_LABELS = {'PAIR_RF': 'Descriptor-RF', 'ECFP_RF': 'ECFP-RF', 'XGB_pair': 'Descriptor-XGB', 'XGB_ecfp': 'ECFP-XGB', 'LGBM_pair': 'Descriptor-LGBM', 'LGBM_ecfp': 'ECFP-LGBM', '3D_pair_RF': 'Guest-shape RF', 'GINE': 'GINE', 'AttentiveFP': 'AttentiveFP'}
MODEL_MARKERS = {'PAIR_RF': 'o', 'ECFP_RF': 's', 'XGB_pair': '^', 'XGB_ecfp': 'v', 'LGBM_pair': 'P', 'LGBM_ecfp': 'X', '3D_pair_RF': 'D', 'GINE': 'h', 'AttentiveFP': '*'}

def model_color(model: str) -> str:
    if model == 'PAIR_RF':
        return RF
    if model == 'GINE':
        return GINE
    if model == 'AttentiveFP':
        return AFP
    return TREE

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def fmt(value: float, digits: int=3, plus: bool=False) -> str:
    spec = f"{('+' if plus else '')}.{digits}f"
    return format(value, spec).replace('-', '−')

def panel_label(fig: plt.Figure, x: float, y: float, label: str, title: str) -> None:
    fig.text(x, y, label, fontsize=10.5, weight='bold', ha='left', va='top')
    fig.text(x + 0.025, y, title, fontsize=9.5, weight='bold', ha='left', va='top')

def clean_axis(ax: plt.Axes, axis: str='x', *, zero: bool=False) -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color(MUTED)
    ax.spines['bottom'].set_color(MUTED)
    ax.spines['left'].set_linewidth(0.65)
    ax.spines['bottom'].set_linewidth(0.65)
    ax.tick_params(width=0.65, length=3.0, pad=3)
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if zero:
        if axis == 'x':
            ax.axvline(0.0, color=MUTED, linewidth=0.85, zorder=1)
        else:
            ax.axhline(0.0, color=MUTED, linewidth=0.85, zorder=1)

def figure3() -> None:
    base_path = DISPLAY / 'Figure_3_display_recipe.json'
    base = read_json(base_path)
    points = base['quantities']['point_values']
    comparisons = base['quantities']['focal_comparisons']
    assert len(points) == 45
    assert len(comparisons) == 9
    values = {(row['model'], row['split']): float(row['pooled_R2_saved_precision']) for row in points}
    sources = {(row['model'], row['split']): row['source'] for row in points}
    splits = ['host_cold', 'guest_cold', 'double_cold']
    split_labels = ['Host-cold', 'Guest-cold', 'Double-cold']
    tree_models = MODEL_ORDER[:7]
    gnn_models = MODEL_ORDER[7:]
    null_models = ['sim_knn_k5', 'additive', 'host_only_RF', 'guest_only_RF', 'cond_only_RF', 'source_only_RF']
    methods = tree_models + gnn_models + null_models
    labels = {**MODEL_LABELS, 'sim_knn_k5': 'Similarity-k5', 'additive': 'Additive', 'host_only_RF': 'Host-only RF', 'guest_only_RF': 'Guest-only RF', 'cond_only_RF': 'Condition+source RF', 'source_only_RF': 'Source-only RF'}
    markers = {**MODEL_MARKERS}
    ypos = {method: float(i) for i, method in enumerate(tree_models)}
    ypos.update({'GINE': 8.0, 'AttentiveFP': 9.0})
    ypos.update({method: float(11 + i) for i, method in enumerate(null_models)})
    best_null = {split: max(null_models, key=lambda model: values[model, split]) for split in splits}
    assert best_null == {'host_cold': 'additive', 'guest_cold': 'sim_knn_k5', 'double_cold': 'guest_only_RF'}
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(WIDTH_MM * MM, 96.0 * MM))
        axes = [fig.add_axes([0.29, 0.145, 0.205, 0.646]), fig.add_axes([0.53, 0.145, 0.205, 0.646]), fig.add_axes([0.77, 0.145, 0.205, 0.646])]
        for panel_i, (ax, split, split_label) in enumerate(zip(axes, splits, split_labels, strict=True)):
            strongest = best_null[split]
            strongest_value = values[strongest, split]
            x0 = ax.get_position().x0
            fig.text(x0, 0.94, f'{chr(97 + panel_i)}  {split_label}', fontsize=9.4, weight='bold', va='top')
            ax.axhspan(7.55, 9.45, facecolor=GNN_BAND, edgecolor='none', zorder=0)
            ax.axhspan(10.55, 16.45, facecolor=NULL_BAND, edgecolor='none', zorder=0)
            ax.axvline(strongest_value, color='#97A4AC', linestyle=(0, (2.3, 2.3)), linewidth=0.8, zorder=1)
            for method in methods:
                value = values[method, split]
                y = ypos[method]
                if method in null_models:
                    if method == strongest:
                        ax.scatter(value, y, s=58, marker='o', facecolor=WHITE, edgecolor=NULL, linewidth=1.15, zorder=5)
                        ax.scatter(value, y, s=19, marker='o', facecolor=WHITE, edgecolor=NULL, linewidth=0.7, zorder=6)
                        offset = (-5, 0) if strongest_value > 0.48 else (5, 0)
                        ax.annotate(fmt(strongest_value), (value, y), xytext=offset, textcoords='offset points', ha='right' if offset[0] < 0 else 'left', va='center', fontsize=8.0, color=NULL, weight='bold')
                    else:
                        ax.scatter(value, y, s=26, marker='o', facecolor=WHITE, edgecolor=NULL, linewidth=0.8, zorder=3)
                else:
                    size = 37 if method == 'PAIR_RF' else 47 if method == 'AttentiveFP' else 31
                    ax.scatter(value, y, s=size, marker=markers[method], facecolor=model_color(method), edgecolor=WHITE, linewidth=0.55, zorder=4)
            for method in ('PAIR_RF', 'GINE', 'AttentiveFP'):
                value = values[method, split]
                y = ypos[method]
                delta = value - strongest_value
                color = model_color(method)
                ax.hlines(y, min(value, strongest_value), max(value, strongest_value), color=color, linewidth=0.8, zorder=2)
                label = fmt(delta, plus=True)
                if split == 'guest_cold' and method == 'PAIR_RF':
                    ax.text(strongest_value - 0.015, y, label, ha='right', va='center', fontsize=8.0, color=color, weight='bold')
                elif split == 'double_cold' and method == 'GINE':
                    ax.text(strongest_value + 0.03, y, label, ha='left', va='center', fontsize=8.0, color=color)
                else:
                    offset = (-4, 0) if value < strongest_value else (-7, 0) if value > 0.6 else (4, 0)
                    ax.annotate(label, (value, y), xytext=offset, textcoords='offset points', ha='right' if offset[0] < 0 else 'left', va='center', fontsize=8.0, color=color, weight='bold' if method == 'PAIR_RF' else 'normal')
            ax.set_xlim(-0.13, 0.73)
            ax.set_xticks([0.0, 0.2, 0.4, 0.6])
            ax.set_ylim(16.75, -1.1)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.spines['bottom'].set_color(MUTED)
            ax.grid(axis='x', color=GRID, linewidth=0.5, zorder=0)
            ax.axvline(0.0, color=MUTED, linewidth=0.9, zorder=1)
            ax.tick_params(axis='y', length=0, pad=5)
            ax.set_yticks([ypos[m] for m in methods])
            if panel_i == 0:
                ax.set_yticklabels([labels[m] for m in methods])
                for y, text_value in [(-0.78, 'TREE MODELS'), (7.3, 'GNNs · 1 INITIALIZATION'), (10.3, 'NULL CONTROLS')]:
                    ax.text(-0.12, y, text_value, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8.0, weight='bold', color=MUTED, clip_on=False)
            else:
                ax.set_yticklabels([])
        fig.text(0.633, 0.055, 'Pooled R²', ha='center', fontsize=8.5)
        save_figure(fig, 'Figure_3')
    assert {(row['model'], row['split']): row['source'] for row in points} == sources

def _group(rows: list[dict[str, str]], **filters: str) -> list[dict[str, str]]:
    return [row for row in rows if all((row.get(key) == value for key, value in filters.items()))]

def figure_s3() -> None:
    base_path = DISPLAY / 'Figure_S3_display_recipe.json'
    csv_path = DISPLAY / 'Figure_S3_display.csv'
    rows = read_csv(csv_path)
    panel_a = _group(rows, panel='a')
    panel_b = _group(rows, panel='b')
    assert len(panel_a) == 40
    assert len(panel_b) == 10
    base = read_json(base_path)
    mean_ticks = base['panel_a']['mean_ticks']
    historical = base['panel_b']['historical_aggregates']
    regimes = ['random', 'host_cold', 'guest_cold', 'double_cold']
    regime_labels = ['Row\nrandom', 'Host\ncold', 'Guest\ncold', 'Double\ncold']
    models = ['GINE', 'AttentiveFP']
    colors = {'GINE': GINE, 'AttentiveFP': AFP}
    markers = {'GINE': 'h', 'AttentiveFP': '*'}
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(WIDTH_MM * MM, 101.0 * MM))
        ax_gine = fig.add_axes([0.08, 0.18, 0.235, 0.583])
        ax_afp = fig.add_axes([0.36, 0.18, 0.235, 0.583])
        ax_b = fig.add_axes([0.73, 0.18, 0.245, 0.583])
        panel_label(fig, 0.025, 0.965, 'a', 'Five training seeds per configuration')
        panel_label(fig, 0.65, 0.965, 'b', 'Released-identifier double-cold\nmargins by seed')
        seed_jitter = np.linspace(-0.16, 0.16, 5)
        for facet_i, (ax, model) in enumerate(zip([ax_gine, ax_afp], models, strict=True)):
            ax.axvspan(2.55, 3.45, color='#F5F1EE', zorder=0)
            ax.axvline(2.5, color=GRID, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
            for ri, regime in enumerate(regimes):
                selected = sorted([row for row in panel_a if row['model'] == model and row['regime'] == regime], key=lambda row: int(row['seed']))
                assert len(selected) == 5
                values = np.asarray([float(row['value']) for row in selected])
                xs = ri + seed_jitter
                ax.scatter(xs, values, s=40 if model == 'GINE' else 58, marker=markers[model], color=colors[model], edgecolor=WHITE, linewidth=0.45, zorder=3)
                mean = float(mean_ticks[model][regime])
                ax.plot([ri - 0.2, ri + 0.2], [mean, mean], color=INK, linewidth=1.3, solid_capstyle='butt', zorder=4)
            ax.axhline(0, color=MUTED, linewidth=0.85, zorder=1)
            ax.set_xticks(range(4), regime_labels)
            ax.tick_params(axis='x', labelsize=8.3, pad=5)
            ax.set_xlim(-0.52, 3.52)
            ax.set_ylim(-0.08, 0.66)
            ax.set_title(model, weight='bold', color=colors[model], pad=8)
            clean_axis(ax, axis='y')
            if facet_i == 0:
                ax.set_ylabel('R²')
            else:
                ax.tick_params(axis='y', left=False, labelleft=False)
        fig.legend(handles=[Line2D([], [], color=INK, linewidth=1.3, label='five-seed mean')], frameon=False, loc='upper left', bbox_to_anchor=(0.28, 0.875), handlelength=2.0)
        offsets = {'GINE': -0.16, 'AttentiveFP': 0.16}
        for model in models:
            selected = sorted([row for row in panel_b if row['model'] == model], key=lambda row: int(row['seed']))
            assert len(selected) == 5
            xs = np.asarray([float(row['value']) for row in selected])
            ys = np.arange(5, dtype=float) + offsets[model]
            ax_b.scatter(xs, ys, s=44 if model == 'GINE' else 62, marker=markers[model], color=colors[model], edgecolor=WHITE, linewidth=0.45, zorder=3)
        ax_b.axvline(float(historical['GINE']), color=GINE, linestyle=(0, (5, 2)), linewidth=1.1, zorder=2)
        ax_b.axvline(float(historical['AttentiveFP']), color=AFP, linestyle=(0, (1.4, 1.7)), linewidth=1.2, zorder=2)
        ax_b.axvline(0, color=MUTED, linewidth=0.9, zorder=1)
        ax_b.set_yticks(range(5), [str(i) for i in range(5)])
        ax_b.set_ylim(4.55, -0.55)
        ax_b.set_xlim(-0.145, 0.015)
        ax_b.set_xlabel('Equal-block mean ΔR²')
        ax_b.set_ylabel('Training seed')
        clean_axis(ax_b, axis='x')
        ax_b.legend(handles=[Line2D([], [], marker='h', linestyle='none', markersize=7, color=GINE, label='GINE'), Line2D([], [], marker='*', linestyle='none', markersize=8, color=AFP, label='AttentiveFP')], frameon=False, ncol=1, loc='lower left', bbox_to_anchor=(0.0, 1.01), handletextpad=0.4)
        ax_b.text(0.97, 0.81, 'AttentiveFP\nhistorical\n−0.046', transform=ax_b.transAxes, fontsize=7.4, linespacing=1.0, color=AFP, ha='right', va='center', bbox={'facecolor': WHITE, 'edgecolor': 'none', 'pad': 1.2})
        ax_b.text(float(historical['GINE']) + 0.003, 4.25, 'GINE historical\n−0.111', fontsize=7.8, color=GINE, ha='left', va='center', bbox={'facecolor': WHITE, 'edgecolor': 'none', 'pad': 1.2})
        save_figure(fig, 'Figure_S3')
