"""Frozen display-only renderer; output handling and metadata adapted."""
from __future__ import annotations
from .common import *
MAPPING = DISPLAY / "Figure_6_display_points.csv"
RECIPE = DISPLAY / "Figure_6_display_recipe.json"

FIGURE_NAME = 'Figure_6'
WIDTH_MM = 165.1
HEIGHT_MM = 160.0
MM = 1.0 / 25.4
PNG_DPI = 600
INK = '#223039'
MUTED = '#51616C'
GRID = '#DDE3E7'
RF = '#226F95'
TREE = '#607887'
GINE = '#885C86'
AFP = '#A3693C'
NULL = '#4F5B64'
SOURCE_NULL = '#7C858B'
ZOOM_FILL = '#EAF1F4'
ZOOM_EDGE = '#8AA4B1'
FAMILIES = ['cyclodextrin', 'cucurbituril', 'calixarene', 'other']
BASES = dict(zip(FAMILIES, [0.0, 11.0, 22.0, 33.0]))
ORDER = ['PAIR_RF', 'ECFP_RF', 'XGB_pair', 'XGB_ecfp', 'LGBM_pair', 'LGBM_ecfp', '3D_pair_RF', 'GINE', 'AttentiveFP']
LABELS = ['Descriptor-RF', 'ECFP-RF', 'Desc.-XGB', 'ECFP-XGB', 'Desc.-LGBM', 'ECFP-LGBM', 'Guest-shape RF', 'GINE', 'AttentiveFP']
MODEL_Y = {model: index for index, model in enumerate(ORDER)}
COLORS = {model: RF if model == 'PAIR_RF' else GINE if model == 'GINE' else AFP if model == 'AttentiveFP' else TREE for model in ORDER}
MARKERS = dict(zip(ORDER, ['o', 's', '^', 'v', 'P', 'X', 'D', 'h', '*']))
SEED_JITTER = {0: -0.32, 1: -0.16, 2: 0.0, 3: 0.16, 4: 0.32}
FULL_R2_LIMITS = (-5.25, 0.78)
DETAIL_R2_LIMITS = (-0.7, 0.42)
RHO_LIMITS = (-0.05, 0.73)
OFFSET_LIMITS = (-2.75, 1.95)
Y_LIMITS = (43.35, -1.25)
PANEL_MAP = {'a': ('a_full_R2', 'guarded_R2'), 'b': ('c_family_pooled_rho', 'rho'), 'c': ('d_observed_minus_predicted_offset', 'offset')}

def style() -> None:
    plt.rcParams.update({'font.family': FONT_FAMILY, 'font.size': 8.1, 'axes.titlesize': 8.8, 'axes.labelsize': 8.1, 'xtick.labelsize': 7.8, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.6, 'axes.linewidth': 0.6, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': MUTED, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.unicode_minus': True, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

def tick_text(value: float, _position: float) -> str:
    if abs(value) < 5e-12:
        value = 0.0
    return f'{value:.1f}'.replace('-', '−')

def load_and_validate() -> tuple[pd.DataFrame, dict[str, tuple[int, int]], list[dict]]:
    frozen = pd.read_csv(MAPPING, dtype=str, float_precision='round_trip', keep_default_na=False)
    required_columns = {'new_panel', 'source_panel', 'source_row_1based_after_header', 'mark_type', 'model', 'family', 'seed', 'metric', 'value', 'unit', 'source', 'is_zoom_repeat', 'display_y', 'numeric_axis_jitter', 'orthogonal_seed_jitter'}
    missing = required_columns.difference(frozen.columns)
    if missing:
        raise RuntimeError(f'Frozen mapping is missing columns: {sorted(missing)}')
    repeat_flag = frozen['is_zoom_repeat'].astype(str).str.lower()
    base = frozen[repeat_flag == 'false'].copy()
    frozen_zoom = frozen[repeat_flag == 'true'].copy()
    if len(base) != 224 or len(frozen_zoom) != 58 or len(frozen) != 282:
        raise RuntimeError(f'Expected frozen counts 224 + 58 = 282, found {len(base)} + {len(frozen_zoom)} = {len(frozen)}')
    pointers = base['source_row_1based_after_header'].astype(int).tolist()
    if pointers != list(range(1, 225)):
        raise RuntimeError('Frozen base-row pointers are not the ordered range 1..224')
    if not np.allclose(frozen['numeric_axis_jitter'].astype(float), 0.0, atol=0.0):
        raise RuntimeError('Frozen mapping changes a numerical-axis coordinate')
    points = base.rename(columns={'source_panel': 'panel'})[['panel', 'mark_type', 'model', 'family', 'seed', 'metric', 'value', 'unit', 'source']].copy().reset_index(drop=True)
    points['seed'] = pd.to_numeric(points['seed'], errors='coerce')
    points['value'] = points['value'].astype(float)
    if set(points['family']) != set(FAMILIES):
        raise RuntimeError('Frozen family set changed')
    expected_roles = {('a', 'supervised_original_point'): 68, ('a', 'guest_only_saved_null'): 4, ('a', 'best_guarded_null_tick'): 4, ('b', 'supervised_original_point'): 68, ('b', 'guest_only_saved_null'): 4, ('c', 'supervised_original_point'): 68, ('c', 'guest_only_saved_null'): 4, ('c', 'source_only_saved_offset'): 4}
    observed_roles = points.groupby(['panel', 'mark_type']).size().to_dict()
    if observed_roles != expected_roles:
        raise RuntimeError(f'Frozen role counts changed: {observed_roles}')
    expected_zoom = base[(base['source_panel'] == 'a') & base['value'].astype(float).between(*DETAIL_R2_LIMITS, inclusive='both')]
    if len(expected_zoom) != 58:
        raise RuntimeError(f'Expected 58 frozen R2 repeats, found {len(expected_zoom)}')
    if frozen_zoom['source_row_1based_after_header'].astype(int).tolist() != expected_zoom['source_row_1based_after_header'].astype(int).tolist():
        raise RuntimeError('Frozen zoom rows do not equal the inclusive in-range subset')
    zoom_by_family = frozen_zoom.groupby('family').size().to_dict()
    if zoom_by_family != {'cyclodextrin': 1, 'cucurbituril': 19, 'calixarene': 19, 'other': 19}:
        raise RuntimeError(f'Unexpected frozen zoom composition: {zoom_by_family}')
    recipe = json.loads(RECIPE.read_text(encoding='utf8'))
    family_counts = {family: (int(recipe['family_counts'][family]['raw_host_identifiers']), int(recipe['family_counts'][family]['released_records'])) for family in FAMILIES}
    return (points, family_counts, frozen.to_dict(orient='records'))

def y_for(row: pd.Series) -> tuple[float, float]:
    base = BASES[str(row['family'])]
    mark_type = str(row['mark_type'])
    if mark_type == 'supervised_original_point':
        model = str(row['model'])
        seed = int(row['seed'])
        jitter = SEED_JITTER[seed] if model in {'GINE', 'AttentiveFP'} else 0.0
        return (base + MODEL_Y[model] + jitter, jitter)
    if mark_type == 'best_guarded_null_tick':
        return (base - 0.52, 0.0)
    if mark_type == 'guest_only_saved_null':
        return (base + 8.95, 0.0)
    if mark_type == 'source_only_saved_offset':
        return (base + 9.32, 0.0)
    raise RuntimeError(f'Unknown mark type: {mark_type}')

def draw_row(ax: plt.Axes, row: pd.Series) -> None:
    value = float(row['value'])
    y, _ = y_for(row)
    mark_type = str(row['mark_type'])
    if mark_type == 'supervised_original_point':
        model = str(row['model'])
        size = 38 if model == 'AttentiveFP' else 28 if model in {'PAIR_RF', 'GINE'} else 23
        ax.scatter([value], [y], s=size, marker=MARKERS[model], color=COLORS[model], edgecolors='white', linewidth=0.35, zorder=5)
    elif mark_type == 'best_guarded_null_tick':
        ax.scatter([value], [y], marker='|', s=86, color='#111820', linewidths=1.65, zorder=7)
    elif mark_type == 'guest_only_saved_null':
        ax.scatter([value], [y], s=31, marker='s', facecolor='white', edgecolor=NULL, linewidth=1.0, zorder=6)
    elif mark_type == 'source_only_saved_offset':
        ax.scatter([value], [y], s=33, marker='D', facecolor='white', edgecolor=SOURCE_NULL, linewidth=1.0, zorder=6)

def configure_axis(ax: plt.Axes, limits: tuple[float, float], ticks: list[float], xlabel: str) -> None:
    ax.set_xlim(*limits)
    ax.set_ylim(*Y_LIMITS)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(tick_text))
    ax.set_xlabel(xlabel, labelpad=6)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.grid(axis='x', color=GRID, linewidth=0.5, zorder=1)
    ax.axvline(0.0, color=MUTED, linewidth=0.8, zorder=2)
    ax.tick_params(axis='y', length=0, pad=3)
    ax.set_axisbelow(True)

def panel_heading(ax: plt.Axes, label: str, title: str, subtitle: str | None=None, *, subtitle_x: float=0.0, subtitle_ha: str='left') -> None:
    ax.text(0.0, 1.064, f'{label}  {title}', transform=ax.transAxes, ha='left', va='bottom', fontsize=8.7, fontweight='bold', clip_on=False)
    if subtitle:
        ax.text(subtitle_x, 1.03, subtitle, transform=ax.transAxes, ha=subtitle_ha, va='bottom', fontsize=7.25, color=MUTED, linespacing=0.95, clip_on=False)

def add_family_structure(axes: list[plt.Axes], fig: plt.Figure) -> None:
    family_label_specs: list[tuple[float, str]] = []
    for family_index, family in enumerate(FAMILIES):
        base = BASES[family]
        shade = '#F4F6F7' if family_index % 2 == 0 else '#FAFBFC'
        for ax in axes:
            ax.axhspan(base - 0.82, base + 9.62, color=shade, zorder=0)
            for yy in base + np.arange(9, dtype=float):
                ax.axhline(yy, color='#E5EAED', linewidth=0.35, zorder=1)
            ax.axhline(base + 6.5, color='#CAD3D8', linewidth=0.65, zorder=1)
        family_label_specs.append((base + 4.15, family))
    panel_box = axes[0].get_position()
    y0, y1 = axes[0].get_ylim()
    for y_value, label in family_label_specs:
        fraction = (y_value - y0) / (y1 - y0)
        figure_y = panel_box.y0 + fraction * panel_box.height
        fig.text(0.012, figure_y, label, ha='left', va='center', fontsize=7.45, color=MUTED)

def append_mapping(mapping_rows: list[dict], row: pd.Series, new_panel: str, source_row: int, is_zoom: bool) -> None:
    display_y, jitter = y_for(row)
    mapping_rows.append({'new_panel': new_panel, 'source_panel': row['panel'], 'source_row_1based_after_header': source_row + 1, 'mark_type': row['mark_type'], 'model': row['model'], 'family': row['family'], 'seed': '' if pd.isna(row['seed']) else int(row['seed']), 'metric': row['metric'], 'value': repr(float(row['value'])), 'unit': row['unit'], 'source': row['source'], 'is_zoom_repeat': str(bool(is_zoom)).lower(), 'display_y': f'{display_y:.4f}', 'numeric_axis_jitter': '0', 'orthogonal_seed_jitter': f'{jitter:.2f}'})

def validate_generated_mapping(mapping_rows: list[dict], frozen_rows: list[dict]) -> None:
    if len(mapping_rows) != len(frozen_rows):
        raise RuntimeError(f'Generated mapping has {len(mapping_rows)} rows; frozen mapping has {len(frozen_rows)}')
    for index, (generated, frozen) in enumerate(zip(mapping_rows, frozen_rows), start=1):
        for field in generated:
            if str(generated[field]) != str(frozen[field]):
                raise RuntimeError(f'Generated mapping row {index}, field {field!r} changed: {generated[field]!r} != {frozen[field]!r}')

def build() -> dict:
    points, family_counts, frozen_mapping = load_and_validate()
    style()
    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))
    grid = fig.add_gridspec(1, 4, width_ratios=[0.78, 1.18, 1.0, 1.08], left=0.302, right=0.992, bottom=0.155, top=0.9, wspace=0.23)
    ax_full = fig.add_subplot(grid[0, 0])
    ax_detail = fig.add_subplot(grid[0, 1])
    ax_rho = fig.add_subplot(grid[0, 2])
    ax_offset = fig.add_subplot(grid[0, 3])
    axes = [ax_full, ax_detail, ax_rho, ax_offset]
    ax_full.axvspan(DETAIL_R2_LIMITS[0], DETAIL_R2_LIMITS[1], facecolor=ZOOM_FILL, edgecolor=ZOOM_EDGE, linewidth=0.65, alpha=0.64, zorder=2.2)
    configure_axis(ax_full, FULL_R2_LIMITS, [-5.0, -2.5, 0.0], '')
    configure_axis(ax_detail, DETAIL_R2_LIMITS, [-0.6, -0.3, 0.0, 0.3], '')
    configure_axis(ax_rho, RHO_LIMITS, [0.0, 0.3, 0.6], '')
    configure_axis(ax_offset, OFFSET_LIMITS, [-2.0, 0.0, 1.5], 'log Kₐ units')
    add_family_structure(axes, fig)
    panel_heading(ax_full, 'a', 'Full-range R²')
    panel_heading(ax_detail, 'b', 'R² zoom', '−0.70 to 0.42')
    panel_heading(ax_rho, 'c', 'Family-pooled ρ')
    panel_heading(ax_offset, 'd', 'Mean offset', 'observed − predicted', subtitle_x=1.0, subtitle_ha='right')
    ax_full.set_yticks(np.arange(9, dtype=float), LABELS)
    ax_detail.tick_params(axis='y', labelleft=False)
    ax_rho.tick_params(axis='y', labelleft=False)
    ax_offset.tick_params(axis='y', labelleft=False)
    mapping_rows: list[dict] = []
    for source_index, row in points.iterrows():
        source_panel = str(row['panel'])
        target_axis = {'a': ax_full, 'b': ax_rho, 'c': ax_offset}[source_panel]
        draw_row(target_axis, row)
        target_panel = PANEL_MAP[source_panel][0]
        append_mapping(mapping_rows, row, target_panel, source_index, False)
        if source_panel == 'a' and DETAIL_R2_LIMITS[0] <= float(row['value']) <= DETAIL_R2_LIMITS[1]:
            draw_row(ax_detail, row)
            append_mapping(mapping_rows, row, 'b_R2_detail', source_index, True)
    handles = [mlines.Line2D([], [], linestyle='none', marker='s', markerfacecolor='white', markeredgecolor=NULL, label='Guest-only RF'), mlines.Line2D([], [], linestyle='none', marker='D', markerfacecolor='white', markeredgecolor=SOURCE_NULL, label='Source-only offset (d)'), mlines.Line2D([], [], linestyle='none', marker='|', markeredgecolor='#111820', markersize=8, label='Best guarded null (R²)')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.615, 0.035), ncol=3, frameon=False, columnspacing=0.85, handletextpad=0.35)
    save_figure(fig, 'Figure_6')
    validate_generated_mapping(mapping_rows, frozen_mapping)
    return {'base_rows': len(points), 'display_roles_including_zoom': len(mapping_rows), 'numeric_axis_jitter': 0, 'frozen_mapping_equal': True}
