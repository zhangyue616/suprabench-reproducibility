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

def _group(rows: list[dict[str, str]], **filters: str) -> list[dict[str, str]]:
    return [row for row in rows if all((row.get(key) == value for key, value in filters.items()))]

def figure4() -> None:
    base_path = DISPLAY / 'Figure_4_display_recipe.json'
    csv_path = DISPLAY / 'Figure_4_display_points.csv'
    base = read_json(base_path)
    rows = read_csv(csv_path)
    counts = Counter(((row['panel'], row['mark_type']) for row in rows))
    assert counts['a', 'matrix_cell'] == 36
    assert counts['b', 'block_point'] == 108
    assert counts['b', 'twelve_block_mean_tick'] == 9
    assert counts['c', 'training_seed_point'] == 45
    assert counts['c', 'five_seed_mean_tick'] == 9
    regimes = ['random', 'host_cold', 'guest_cold', 'double_cold']
    regime_labels = ['Row-\nrandom', 'Host-\ncold', 'Guest-\ncold', 'Double-\ncold†']
    matrix_lookup = {(row['model'], row['regime']): float(row['value']) for row in _group(rows, panel='a', mark_type='matrix_cell')}
    matrix = np.asarray([[matrix_lookup[model, regime] for regime in regimes] for model in MODEL_ORDER], dtype=float)
    five_seed_cells = {(model, regime) for model in MODEL_ORDER for regime in regimes if model in {'GINE', 'AttentiveFP'} or regime == 'double_cold'}
    single_run_cells = set(matrix_lookup) - five_seed_cells
    assert len(five_seed_cells) == 15
    assert len(single_run_cells) == 21
    positive_blocks = base['panel_c_positive_blocks_per_seed']
    assert set(positive_blocks) == set(MODEL_ORDER)
    cmap = LinearSegmentedColormap.from_list('seed_blues', ['#EEF5F8', '#69A9CC', '#0D579C'])
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(WIDTH_MM * MM, 145.5 * MM))
        ax_a = fig.add_axes([0.18, 0.585, 0.335, 0.305])
        ax_b = fig.add_axes([0.61, 0.585, 0.365, 0.305])
        ax_c = fig.add_axes([0.18, 0.105, 0.585, 0.285])
        ax_ct = fig.add_axes([0.79, 0.105, 0.19, 0.285], sharey=ax_c)
        panel_label(fig, 0.18, 0.965, 'a', 'R² score context')
        fig.text(0.18, 0.915, '† rows/column: five-seed means · otherwise: single runs', fontsize=8.0, color=MUTED, ha='left', va='top')
        panel_label(fig, 0.61, 0.965, 'b', 'Block margins')
        ax_a.imshow(matrix, cmap=cmap, vmin=0.2, vmax=0.7, aspect='auto', interpolation='none')
        ax_a.set_xticks(range(4), regime_labels)
        row_labels = [MODEL_LABELS[model] + ('†' if model in {'GINE', 'AttentiveFP'} else '') for model in MODEL_ORDER]
        ax_a.set_yticks(range(9), row_labels)
        ax_a.tick_params(length=0)
        for i in range(9):
            for j in range(4):
                value = matrix[i, j]
                color = WHITE if value >= 0.52 else INK
                ax_a.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=8.3, color=color)
        for spine in ax_a.spines.values():
            spine.set_visible(False)
        ymap = {model: float(i) for i, model in enumerate(MODEL_ORDER)}
        block_jitter = np.linspace(-0.22, 0.22, 12)
        for model in MODEL_ORDER:
            block_rows = sorted(_group(rows, panel='b', mark_type='block_point', model=model), key=lambda row: int(row['block']))
            assert len(block_rows) == 12
            xs = np.asarray([float(row['value']) for row in block_rows])
            ys = ymap[model] + block_jitter
            ax_b.scatter(xs, ys, s=29 if model != 'AttentiveFP' else 40, marker=MODEL_MARKERS[model], color=model_color(model), edgecolor=WHITE, linewidth=0.45, zorder=3)
            mean_row = _group(rows, panel='b', mark_type='twelve_block_mean_tick', model=model)
            assert len(mean_row) == 1
            mean = float(mean_row[0]['value'])
            ax_b.plot([mean, mean], [ymap[model] - 0.28, ymap[model] + 0.28], color=INK, linewidth=1.2, solid_capstyle='butt', zorder=4)
        ax_b.axvline(0, color=MUTED, linewidth=0.85, zorder=1)
        ax_b.set_xlim(-0.7, 0.42)
        ax_b.set_ylim(8.65, -0.65)
        ax_b.set_yticks([])
        ax_b.set_xlabel('Block-level margin ΔR²')
        clean_axis(ax_b, axis='x')
        panel_label(fig, 0.18, 0.465, 'c', 'Training-seed margins')
        seed_jitter = np.linspace(-0.18, 0.18, 5)
        mean_values: dict[str, float] = {}
        for model in MODEL_ORDER:
            seed_rows = sorted(_group(rows, panel='c', mark_type='training_seed_point', model=model), key=lambda row: int(row['seed']))
            assert len(seed_rows) == 5
            xs = np.asarray([float(row['value']) for row in seed_rows])
            ys = ymap[model] + seed_jitter
            ax_c.scatter(xs, ys, s=31 if model != 'AttentiveFP' else 44, marker=MODEL_MARKERS[model], color=model_color(model), edgecolor=WHITE, linewidth=0.45, zorder=3)
            mean_row = _group(rows, panel='c', mark_type='five_seed_mean_tick', model=model)
            assert len(mean_row) == 1
            mean = float(mean_row[0]['value'])
            mean_values[model] = mean
            ax_c.plot([mean, mean], [ymap[model] - 0.29, ymap[model] + 0.29], color=INK, linewidth=1.2, solid_capstyle='butt', zorder=4)
        ax_c.axvline(0, color=MUTED, linewidth=0.85, zorder=1)
        ax_c.set_xlim(-0.15, 0.12)
        ax_c.set_ylim(8.65, -0.65)
        ax_c.set_yticks(range(9), [MODEL_LABELS[m] for m in MODEL_ORDER])
        ax_c.set_xlabel('Training-seed mean margin ΔR²')
        clean_axis(ax_c, axis='x')
        ax_ct.set_xlim(0, 1)
        ax_ct.set_ylim(8.65, -0.65)
        ax_ct.axis('off')
        ax_ct.text(0.0, -0.92, 'mean ΔR² | positive\nblocks per seed', fontsize=7.7, weight='bold', linespacing=1.1, ha='left', va='bottom', clip_on=False)
        for model in MODEL_ORDER:
            ax_ct.text(0.0, ymap[model], f'{fmt(mean_values[model], plus=True)}  |  {positive_blocks[model]}', fontsize=8.1, ha='left', va='center')
        save_figure(fig, 'Figure_4')

def figure5() -> None:
    csv_path = DISPLAY / 'Figure_5_display_points.csv'
    rows = read_csv(csv_path)
    assert len(_group(rows, panel='a', mark_type='paired_point')) == 12
    assert len(_group(rows, panel='b', mark_type='condition_point')) == 3
    assert len(_group(rows, panel='c', mark_type='paired_point')) == 6
    height_mm = 120.0
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(WIDTH_MM * MM, height_mm * MM))
        ax_r2 = fig.add_axes([0.17, 67.2 / height_mm, 0.35, 28.896 / height_mm])
        ax_mae = fig.add_axes([0.585, 67.2 / height_mm, 0.365, 28.896 / height_mm])
        ax_b = fig.add_axes([0.17, 13.44 / height_mm, 0.305, 33.6 / height_mm])
        ax_c = fig.add_axes([0.685, 13.44 / height_mm, 0.265, 33.6 / height_mm])
        legend_ax = fig.add_axes([0.17, 0.82, 0.78, 0.065])
        legend_ax.set_axis_off()
        panel_label(fig, 0.025, 0.965, 'a', 'Similarity strata')
        regimes = ['s < 0.4', '0.4 <= s <= 0.7', 's > 0.7']
        display_regimes = ['s < 0.4\nn=319', '0.4 ≤ s ≤ 0.7\nn=1,211', 's > 0.7\nn=454']
        y = np.asarray([2.0, 1.0, 0.0])
        panel_a = _group(rows, panel='a', mark_type='paired_point')
        for ax, metric, xlim, xlabel in [(ax_r2, 'R2', (-0.34, 0.58), 'Pooled-bin R²'), (ax_mae, 'MAE', (1.02, 1.62), 'Pooled-bin MAE (log $K_{\\mathrm{a}}$)')]:
            for yi, regime in zip(y, regimes, strict=True):
                rf_row = [r for r in panel_a if r['regime'] == regime and r['metric'] == metric and (r['model'] == 'PAIR_RF')]
                k_row = [r for r in panel_a if r['regime'] == regime and r['metric'] == metric and (r['model'] == 'sim_knn_k1')]
                assert len(rf_row) == len(k_row) == 1
                rf_value = float(rf_row[0]['value'])
                k_value = float(k_row[0]['value'])
                ax.plot([rf_value, k_value], [yi, yi], color=CONNECTOR, linewidth=1.0, zorder=1)
                ax.scatter(rf_value, yi, s=43, marker='o', color=RF, edgecolor=WHITE, linewidth=0.5, zorder=3)
                ax.scatter(k_value, yi, s=45, marker='s', color=AFP, edgecolor=WHITE, linewidth=0.5, zorder=3)
            ax.set_xlim(*xlim)
            ax.set_ylim(-0.5, 2.5)
            ax.set_xlabel(xlabel)
            clean_axis(ax, axis='x', zero=metric == 'R2')
        ax_r2.set_yticks(y, display_regimes)
        ax_mae.set_yticks([])
        handles = [Line2D([], [], marker='o', linestyle='none', markersize=6.5, color=RF, label='Descriptor-RF'), Line2D([], [], marker='s', linestyle='none', markersize=6.5, color=AFP, label='Similarity k1')]
        panel_a_legend = legend_ax.legend(handles=handles, frameon=False, ncol=2, loc='center left', bbox_to_anchor=(0.0, 0.5), borderaxespad=0.0, handletextpad=0.5, columnspacing=1.2)
        lower_title_y = 56.0 / height_mm
        panel_label(fig, 0.17, lower_title_y, 'b', 'Recorded condition/source fields')
        b_rows = _group(rows, panel='b', mark_type='condition_point')
        b_order = ['Condition-only RF', 'Descriptor-RF', 'Descriptor-RF + conditions']
        b_labels = ['Condition + source', 'Descriptor-RF', 'RF + fields']
        b_markers = ['s', 'o', 'D']
        b_colors = [NULL, RF, '#6EA6C5']
        by = np.asarray([2.0, 1.0, 0.0])
        b_values = []
        for yi, model, marker, color in zip(by, b_order, b_markers, b_colors, strict=True):
            selected = [row for row in b_rows if row['model'] == model]
            assert len(selected) == 1
            value = float(selected[0]['value'])
            b_values.append(value)
            ax_b.scatter(value, yi, s=45 if marker != 'D' else 52, marker=marker, color=color, edgecolor=WHITE, linewidth=0.5, zorder=3)
            if value < 0:
                ax_b.annotate(fmt(value), (value, yi), xytext=(-0.025, yi + 0.24), textcoords='data', ha='right', va='bottom', fontsize=8.3, bbox={'facecolor': WHITE, 'edgecolor': 'none', 'pad': 0.8}, arrowprops={'arrowstyle': '-', 'color': CONNECTOR, 'linewidth': 0.8, 'shrinkA': 2, 'shrinkB': 4})
            else:
                ax_b.annotate(fmt(value), (value, yi), xytext=(6, 0), textcoords='offset points', ha='left', va='center', fontsize=8.3)
        ax_b.plot([b_values[1], b_values[2]], [by[1], by[2]], color=CONNECTOR, linewidth=1.0, zorder=1)
        ax_b.set_yticks(by, b_labels)
        ax_b.set_ylim(-0.55, 2.55)
        ax_b.set_xlim(-0.18, 0.44)
        ax_b.set_xlabel('12-block equal-weight mean R²')
        clean_axis(ax_b, axis='x', zero=True)
        panel_label(fig, 0.685, lower_title_y, 'c', 'Low-similarity family labels')
        c_rows = _group(rows, panel='c', mark_type='paired_point')
        families = ['calixarene', 'other', 'cyclodextrin']
        family_labels = ['calixarene\nn=53', 'other\nn=155', 'cyclodextrin\nn=70']
        cy = np.asarray([2.0, 1.0, 0.0])
        for yi, family in zip(cy, families, strict=True):
            rf_row = [r for r in c_rows if r['family'] == family and r['model'] == 'PAIR_RF']
            k_row = [r for r in c_rows if r['family'] == family and r['model'] == 'sim_knn_k5']
            assert len(rf_row) == len(k_row) == 1
            rf_value = float(rf_row[0]['value'])
            k_value = float(k_row[0]['value'])
            ax_c.plot([rf_value, k_value], [yi, yi], color=CONNECTOR, linewidth=1.0, zorder=1)
            ax_c.scatter(rf_value, yi, s=43, marker='o', color=RF, edgecolor=WHITE, linewidth=0.5, zorder=3)
            ax_c.scatter(k_value, yi, s=45, marker='^', color=AFP, edgecolor=WHITE, linewidth=0.5, zorder=3)
        ax_c.set_yticks(cy, family_labels)
        ax_c.set_ylim(-0.55, 2.55)
        ax_c.set_xlim(-0.38, 0.46)
        ax_c.set_xlabel('Pooled low-sim bin R²')
        clean_axis(ax_c, axis='x', zero=True)
        fig.legend(handles=[Line2D([], [], marker='o', linestyle='none', markersize=6.5, color=RF, label='Descriptor-RF'), Line2D([], [], marker='^', linestyle='none', markersize=7.0, color=AFP, label='Similarity k5')], frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0.69, 52.416 / height_mm), handletextpad=0.35, columnspacing=0.75)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = panel_a_legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        panel_a_axes_top = max(ax_r2.get_position().y1, ax_mae.get_position().y1)
        legend_gap_fraction = legend_bbox.y0 - panel_a_axes_top
        legend_gap_mm = legend_gap_fraction * height_mm
        assert legend_gap_mm > 1.0
        save_figure(fig, 'Figure_5')
