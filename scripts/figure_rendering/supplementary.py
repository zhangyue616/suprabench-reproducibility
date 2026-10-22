"""Frozen display-only renderer; output handling and metadata adapted."""
from __future__ import annotations
from .common import *

MM = 1.0 / 25.4
WIDTH_MM = 165.1
INK = '#223039'
MUTED = '#51616C'
GRID = '#DDE3E7'
RF = '#226F95'
AFP = '#A3693C'
NULL = '#4F5B64'
WHITE = '#FFFFFF'
RC = {'font.family': FONT_FAMILY, 'font.sans-serif': [FONT_FAMILY], 'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8.2, 'ytick.labelsize': 8.2, 'legend.fontsize': 8.0, 'axes.linewidth': 0.65, 'lines.linewidth': 0.9, 'patch.linewidth': 0.6, 'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': MUTED, 'xtick.color': INK, 'ytick.color': INK, 'axes.unicode_minus': True, 'figure.facecolor': WHITE, 'axes.facecolor': WHITE, 'savefig.facecolor': WHITE, 'savefig.edgecolor': 'none', 'svg.fonttype': 'none', 'pdf.fonttype': 42}

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

def clean_axis(ax: plt.Axes, axis: str='x') -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color(MUTED)
    ax.spines['bottom'].set_color(MUTED)
    ax.spines['left'].set_linewidth(0.65)
    ax.spines['bottom'].set_linewidth(0.65)
    ax.tick_params(width=0.65, length=3.0, pad=3)
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

def _group(rows: list[dict[str, str]], **filters: str) -> list[dict[str, str]]:
    return [row for row in rows if all((row[key] == value for key, value in filters.items()))]

def _bbox_points(artist, renderer, fig: plt.Figure) -> dict[str, float]:
    bbox = artist.get_window_extent(renderer=renderer)
    scale = 72.0 / fig.dpi
    return {'x0': round(bbox.x0 * scale, 3), 'y0': round(bbox.y0 * scale, 3), 'x1': round(bbox.x1 * scale, 3), 'y1': round(bbox.y1 * scale, 3), 'width': round(bbox.width * scale, 3), 'height': round(bbox.height * scale, 3)}

def _axes_bbox_points(ax: plt.Axes, renderer, fig: plt.Figure) -> dict[str, float]:
    bbox = ax.get_window_extent(renderer=renderer)
    scale = 72.0 / fig.dpi
    return {'x0': round(bbox.x0 * scale, 3), 'y0': round(bbox.y0 * scale, 3), 'x1': round(bbox.x1 * scale, 3), 'y1': round(bbox.y1 * scale, 3)}

def figure_s1() -> dict:
    csv_path = DISPLAY / 'Figure_S1_display.csv'
    rows = read_csv(csv_path)
    a_rows = _group(rows, panel='a')
    b_rows = _group(rows, panel='b')
    c_rows = _group(rows, panel='c')
    assert len(a_rows) == 15
    assert len(b_rows) == 6
    assert len(c_rows) == 2
    assert all((int(row['denominator']) == 2383 for row in a_rows))
    group_order = ['source', 'solvent', 'temperature', 'pH']
    group_titles = ['Source', 'Solvent', 'Temperature', 'pH']
    group_labels = {'source': {'eupmc': 'eupmc', 'cb7_supplement': 'cb7\nsupplement'}, 'solvent': {'buffer': 'buffer', 'water': 'water', 'complex': 'complex', 'D2O': 'D$_2$O', 'missing': 'missing', 'other': 'other'}, 'temperature': {'~25': '~25 °C', 'missing': 'missing', 'other': 'other'}, 'pH': {'missing': 'missing', 'neutral': 'neutral', 'acidic': 'acidic', 'basic': 'basic'}}
    fills = {'source': '#5F7B8A', 'solvent': '#7894A2', 'temperature': '#91A6B0', 'pH': '#A7B7BF'}
    positions = [[0.105, 0.585, 0.17, 0.275], [0.365, 0.585, 0.185, 0.275], [0.64, 0.585, 0.155, 0.275], [0.855, 0.585, 0.12, 0.275]]
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(WIDTH_MM * MM, 121.4 * MM))
        panel_label(fig, 0.025, 0.972, 'a', 'Released-field coverage')
        solvent_text_artists = []
        solvent_text_strings = []
        for group, title, pos in zip(group_order, group_titles, positions, strict=True):
            ax = fig.add_axes(pos)
            selected = [row for row in a_rows if row['group'] == group]
            y = np.arange(len(selected))[::-1]
            values = np.asarray([float(row['value']) for row in selected])
            ax.barh(y, values, color=fills[group], edgecolor=WHITE, linewidth=0.5, height=0.7, zorder=2)
            ax.set_yticks(y, [group_labels[group][row['category']] for row in selected])
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100], ['0', '50', '100%'])
            ax.set_title(title, loc='left', weight='bold', pad=6)
            clean_axis(ax, axis='x')
            for yi, row, value in zip(y, selected, values, strict=True):
                if group == 'solvent':
                    text = f"{int(row['count']):,} ({value:.1f}%)"
                    artist = ax.text(value + 1.8, yi, text, ha='left', va='center', fontsize=8.1, color=INK, zorder=4)
                    solvent_text_artists.append(artist)
                    solvent_text_strings.append(text)
                else:
                    text = f"{int(row['count']):,}\n{value:.1f}%"
                    inside = group == 'source' and value > 60
                    x = value - 2.2 if inside else min(value + 2.0, 96.5)
                    ax.text(x, yi, text, ha='right' if inside else 'left', va='center', fontsize=8.1, color=WHITE if inside else INK, linespacing=1.25, zorder=4)
        panel_label(fig, 0.105, 0.47, 'b', 'Recorded solvent strata')
        panel_label(fig, 0.7, 0.47, 'c', 'Released-record dispersion')
        ax_b = fig.add_axes([0.105, 0.105, 0.455, 0.285])
        ax_c = fig.add_axes([0.7, 0.105, 0.275, 0.285])
        solvents = ['buffer', 'water', 'complex']
        solvent_labels = ['buffer\n12 blocks', 'water\n8 blocks', 'complex\n8 blocks']
        sy = np.asarray([2.0, 1.0, 0.0])
        for yi, solvent in zip(sy, solvents, strict=True):
            rf_row = [r for r in b_rows if r['category'] == solvent and r['series'] == 'Descriptor-RF']
            k_row = [r for r in b_rows if r['category'] == solvent and r['series'] == 'Similarity k=5']
            assert len(rf_row) == len(k_row) == 1
            rf_value = float(rf_row[0]['value'])
            rf_sd = float(rf_row[0]['uncertainty'])
            k_value = float(k_row[0]['value'])
            ax_b.errorbar(rf_value, yi + 0.12, xerr=rf_sd, fmt='o', markersize=6.2, color=RF, ecolor=RF, elinewidth=1.0, capsize=2.5, markeredgecolor=WHITE, markeredgewidth=0.5, zorder=3)
            ax_b.scatter(k_value, yi - 0.12, s=49, marker='^', color=AFP, edgecolor=WHITE, linewidth=0.5, zorder=4)
            if solvent == 'complex':
                ax_b.annotate(fmt(k_value), (k_value, yi - 0.12), xytext=(5, 8), textcoords='offset points', ha='left', va='bottom', fontsize=8.1)
        ax_b.axvline(0, color=MUTED, linewidth=0.85, zorder=1)
        ax_b.set_xlim(-1.2, 0.66)
        ax_b.set_ylim(-0.55, 2.55)
        ax_b.set_yticks(sy, solvent_labels)
        ax_b.set_xlabel('Within-solvent double-cold R²')
        clean_axis(ax_b, axis='x')
        ax_b.legend(handles=[Line2D([], [], marker='^', linestyle='none', markersize=7, color=AFP, label='Similarity k=5 mean'), Line2D([], [], marker='o', linestyle='-', markersize=6, color=RF, label='Descriptor-RF mean ± block SD')], frameon=False, ncol=2, loc='lower left', bbox_to_anchor=(0.0, 1.02), handletextpad=0.4, columnspacing=1.0)
        c_labels = ['Same-condition groups · n=75', 'Cross-condition pairs · n=47']
        cy = np.asarray([1.0, 0.0])
        c_colors = [NULL, '#98A7B0']
        c_label_artists = []
        for yi, row, color, label in zip(cy, c_rows, c_colors, c_labels, strict=True):
            value = float(row['value'])
            ax_c.hlines(yi, 0, value, color=color, linewidth=1.1, zorder=2)
            ax_c.scatter(value, yi, s=42, color=color, edgecolor=WHITE, linewidth=0.5, zorder=3)
            ax_c.annotate(fmt(value), (value, yi), xytext=(4, 0), textcoords='offset points', ha='left', va='center', fontsize=8.1)
            c_label_artists.append(ax_c.text(0.006, yi + 0.17, label, ha='left', va='bottom', fontsize=8.1, color=INK, zorder=4))
        ax_c.set_yticks(cy, ['', ''])
        ax_c.tick_params(axis='y', length=0)
        ax_c.set_ylim(-0.5, 1.5)
        ax_c.set_xlim(0, 0.285)
        ax_c.set_xlabel('Mean groupwise sample SD\n(log $K_{\\mathrm{a}}$)')
        clean_axis(ax_c, axis='x')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        solvent_bboxes = [_bbox_points(artist, renderer, fig) for artist in solvent_text_artists]
        sorted_solvent_bboxes = sorted(solvent_bboxes, key=lambda box: box['y0'])
        solvent_gaps = [round(sorted_solvent_bboxes[index + 1]['y0'] - sorted_solvent_bboxes[index]['y1'], 3) for index in range(len(sorted_solvent_bboxes) - 1)]
        b_axes_bbox = _axes_bbox_points(ax_b, renderer, fig)
        c_label_bboxes = [_bbox_points(artist, renderer, fig) for artist in c_label_artists]
        c_label_gap = min((box['x0'] for box in c_label_bboxes)) - b_axes_bbox['x1']
        placement_scale = 161.5 / WIDTH_MM
        diagnostics = {'baseline': 'retained display layout', 'native_width_mm': WIDTH_MM, 'placement_width_mm': 161.5, 'panel_a_solvent_labels': solvent_text_strings, 'panel_a_solvent_label_count': len(solvent_text_strings), 'panel_a_min_adjacent_vertical_glyph_gap_pt_native': round(min(solvent_gaps), 3), 'panel_a_min_adjacent_vertical_glyph_gap_pt_at_161_5_mm': round(min(solvent_gaps) * placement_scale, 3), 'panel_c_labels': c_labels, 'panel_c_label_to_panel_b_data_gap_pt_native': round(c_label_gap, 3), 'panel_c_label_to_panel_b_data_gap_pt_at_161_5_mm': round(c_label_gap * placement_scale, 3), 'panel_b_xlim': [float(value) for value in ax_b.get_xlim()], 'assertions': {'six_solvent_labels_are_single_line': bool(len(solvent_text_strings) == 6 and all(('\n' not in text for text in solvent_text_strings))), 'solvent_glyphs_do_not_overlap_vertically': bool(min(solvent_gaps) > 0), 'panel_c_labels_do_not_intrude_into_panel_b_data': bool(c_label_gap > 0), 'panel_b_axis_range_unchanged': bool(tuple((round(float(value), 2) for value in ax_b.get_xlim())) == (-1.2, 0.66))}}
        assert all(diagnostics['assertions'].values())
        save_figure(fig, 'Figure_S1')
    return diagnostics

def figure_s2() -> dict:
    csv_path = DISPLAY / 'Figure_S2_display.csv'
    rows = read_csv(csv_path)
    pooled = _group(rows, panel='a', section='pooled')
    block = _group(rows, panel='a', section='equal_block_double_cold')
    slopes = _group(rows, panel='b', section='family_slope')
    assert len(pooled) == 9
    assert len(block) == 3
    assert len(slopes) == 16
    k_order = ['k=1', 'k=3', 'k=5']
    k_markers = {'k=1': 's', 'k=3': 'o', 'k=5': '^'}
    k_offsets = {'k=1': 0.22, 'k=3': 0.0, 'k=5': -0.22}
    regimes = ['random', 'host-cold', 'guest-cold']
    regime_labels = ['Row-random', 'Host-cold', 'Guest-cold']
    ry = {'random': 2.0, 'host-cold': 1.0, 'guest-cold': 0.0}
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(WIDTH_MM * MM, 105.0 * MM))
        panel_label(fig, 0.025, 0.97, 'a', 'Similarity-retrieval neighborhood size')
        ax_pool = fig.add_axes([0.13, 0.17, 0.25, 0.585])
        ax_block = fig.add_axes([0.445, 0.17, 0.195, 0.585])
        ax_b = fig.add_axes([0.785, 0.17, 0.19, 0.585])
        pool_value_artists = []
        for row in pooled:
            regime = row['category']
            series = row['series']
            value = float(row['value'])
            y = ry[regime] + k_offsets[series]
            ax_pool.scatter(value, y, s=44 if series != 'k=3' else 47, marker=k_markers[series], color=AFP, edgecolor=WHITE, linewidth=0.5, zorder=3)
            pool_value_artists.append(ax_pool.annotate(fmt(value), (value, y), xytext=(4, 0), textcoords='offset points', ha='left', va='center', fontsize=8.1))
        ax_pool.axvline(0, color=MUTED, linewidth=0.85, zorder=1)
        ax_pool.set_xlim(-0.05, 0.7)
        ax_pool.set_ylim(-0.52, 2.52)
        ax_pool.set_yticks([2, 1, 0], regime_labels)
        ax_pool.set_xticks([0.0, 0.2, 0.4, 0.6])
        ax_pool.set_xlabel('Pooled R²')
        fig.text(0.13, 0.825, 'Pooled · 3 regimes', fontsize=9.5, weight='bold', ha='left', va='top')
        clean_axis(ax_pool, axis='x')
        shape_legend = fig.legend(handles=[Line2D([], [], marker=k_markers[k], linestyle='none', markersize=6.5, color=AFP, label=k) for k in k_order], frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0.13, 0.885), handletextpad=0.3, columnspacing=0.75)
        by = {'k=1': 2.0, 'k=3': 1.0, 'k=5': 0.0}
        for row in block:
            series = row['series']
            value = float(row['value'])
            sd = float(row['sd'])
            y = by[series]
            ax_block.errorbar(value, y, xerr=sd, fmt=k_markers[series], markersize=6.2, color=AFP, ecolor=AFP, elinewidth=1.0, capsize=2.4, markeredgecolor=WHITE, markeredgewidth=0.5, zorder=3)
            ax_block.annotate(f'{fmt(value, plus=True)} ± {sd:.3f}', (value + sd, y), xytext=(4, 0), textcoords='offset points', ha='left', va='center', fontsize=7.5)
        ax_block.axvline(0, color=MUTED, linewidth=0.85, zorder=1)
        ax_block.set_xlim(-0.58, 1.35)
        ax_block.set_xticks([-0.5, 0.0, 0.5, 1.0])
        ax_block.set_ylim(-0.52, 2.52)
        ax_block.set_yticks([2, 1, 0], ['', '', ''])
        ax_block.tick_params(axis='y', length=0)
        ax_block.set_xlabel('Equal-block mean R²')
        fig.text(0.445, 0.885, 'Double-cold\nmean ± SD · 12 blocks', fontsize=9.5, weight='bold', linespacing=1.05, ha='left', va='top')
        clean_axis(ax_block, axis='x')
        supported_families = ['cucurbituril', 'other', 'calixarene', 'cyclodextrin']
        small_families = ['cavitand', 'pillararene', 'cyclophane', 'crown ether']
        family_order = supported_families + small_families
        fy = {family: float(i if i < 4 else i + 1) for i, family in enumerate(family_order)}
        ax_b.axhspan(-0.5, 3.5, color='#F4F7F8', zorder=0)
        for family in family_order:
            pair = [r for r in slopes if r['category'] == family and r['series'] == 'Descriptor-RF']
            afp = [r for r in slopes if r['category'] == family and r['series'] == 'AttentiveFP']
            assert len(pair) == len(afp) == 1
            pair_value = float(pair[0]['value'])
            afp_value = float(afp[0]['value'])
            ok = pair[0]['support'] == 'supported'
            y = fy[family]
            ax_b.plot([pair_value, afp_value], [y, y], color='#C8D0D5', linewidth=1.0, zorder=1)
            ax_b.scatter(pair_value, y, s=42, marker='o', facecolor=RF if ok else WHITE, edgecolor=RF, linewidth=1.0, zorder=3)
            ax_b.scatter(afp_value, y, s=54, marker='*', facecolor=AFP if ok else WHITE, edgecolor=AFP, linewidth=1.0, zorder=3)
        ax_b.axvline(1.0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
        ax_b.axhline(4.0, color=GRID, linewidth=0.8, zorder=1)
        ax_b.set_yticks([fy[family] for family in family_order], family_order)
        ax_b.set_ylim(8.55, -0.55)
        ax_b.set_xlim(-0.2, 1.82)
        ax_b.set_xlabel('OLS slope\n(observed on predicted)', fontsize=7.8)
        fig.text(0.75, 0.97, 'b  Historical original-run\nfamily slopes', fontsize=9.5, weight='bold', linespacing=1.05, ha='left', va='top')
        clean_axis(ax_b, axis='x')
        ax_b.text(0.98, 1.01, 'supported', transform=ax_b.transAxes, ha='right', va='bottom', fontsize=8.2, color=MUTED)
        small_support_artist = ax_b.text(1.1, 4.5, 'small\nsupport', ha='left', va='center', fontsize=8.2, linespacing=0.8, color=MUTED, zorder=4)
        ax_b.legend(handles=[Line2D([], [], marker='o', linestyle='none', markersize=6.5, color=RF, label='Descriptor-RF'), Line2D([], [], marker='*', linestyle='none', markersize=8.0, color=AFP, label='AttentiveFP')], frameon=False, ncol=1, loc='lower left', bbox_to_anchor=(-0.03, 1.01), handletextpad=0.35)
        association = {category: {row['series']: float(row['value']) for row in pooled if row['category'] == category} for category in regimes}
        expected_association = {'random': {'k=1': 0.481, 'k=3': 0.582, 'k=5': 0.587}, 'host-cold': {'k=1': 0.095, 'k=3': 0.239, 'k=5': 0.218}, 'guest-cold': {'k=1': 0.362, 'k=3': 0.505, 'k=5': 0.538}}
        assert association == expected_association
        expected_block = {'k=1': {'mean': -0.123, 'sd': 0.372}, 'k=3': {'mean': 0.134, 'sd': 0.331}, 'k=5': {'mean': 0.186, 'sd': 0.305}}
        actual_block = {row['series']: {'mean': float(row['value']), 'sd': float(row['sd'])} for row in block}
        assert actual_block == expected_block
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        pool_value_bboxes = [_bbox_points(artist, renderer, fig) for artist in pool_value_artists]
        block_axes_bbox = _axes_bbox_points(ax_block, renderer, fig)
        pool_to_block_gap = block_axes_bbox['x0'] - max((box['x1'] for box in pool_value_bboxes))
        small_support_bbox = _bbox_points(small_support_artist, renderer, fig)
        panel_b_axes_bbox = _axes_bbox_points(ax_b, renderer, fig)
        zero_grid_x_pt = float(ax_b.transData.transform((0.0, 4.0))[0] * 72.0 / fig.dpi)
        slope_one_x_pt = float(ax_b.transData.transform((1.0, 4.0))[0] * 72.0 / fig.dpi)
        divider_y_pt = float(ax_b.transData.transform((0.0, 4.0))[1] * 72.0 / fig.dpi)
        first_small_row_y_pt = float(ax_b.transData.transform((0.0, 5.0))[1] * 72.0 / fig.dpi)
        small_support_clearances = {'from_left_axis_spine_pt': round(small_support_bbox['x0'] - panel_b_axes_bbox['x0'], 3), 'from_x0_gridline_pt': round(small_support_bbox['x0'] - zero_grid_x_pt, 3), 'from_slope1_reference_line_pt': round(small_support_bbox['x0'] - slope_one_x_pt, 3), 'to_right_axis_boundary_pt': round(panel_b_axes_bbox['x1'] - small_support_bbox['x1'], 3), 'below_y4_group_divider_pt': round(divider_y_pt - small_support_bbox['y1'], 3), 'above_first_small_support_row_pt': round(small_support_bbox['y0'] - first_small_row_y_pt, 3)}
        legend_labels = [text.get_text() for text in shape_legend.get_texts()]
        visible_block_tick_labels = [text.get_text() for text in ax_block.get_yticklabels() if text.get_text().strip()]
        diagnostics = {'baseline': 'retained display layout', 'native_width_mm': WIDTH_MM, 'placement_width_mm': WIDTH_MM, 'panel_a_shape_legend_labels': legend_labels, 'panel_a_right_repeated_k_tick_labels': visible_block_tick_labels, 'panel_a_pool_value_to_double_cold_data_gap_pt': round(pool_to_block_gap, 3), 'panel_b_small_support_label': 'small\nsupport', 'panel_b_small_support_data_coordinates': [1.1, 4.5], 'panel_b_small_support_clearances': small_support_clearances, 'panel_a_exact_marker_value_association': expected_association, 'panel_a_double_cold_mean_sd': expected_block, 'assertions': {'shape_legend_retains_k1_k3_k5': bool(legend_labels == k_order), 'right_repeated_k_tick_labels_removed': bool(visible_block_tick_labels == []), 'pooled_value_labels_end_before_double_cold_data_axis': bool(pool_to_block_gap > 0), 'small_support_clear_of_left_axis_spine': bool(small_support_clearances['from_left_axis_spine_pt'] > 0), 'small_support_clear_of_x0_gridline': bool(small_support_clearances['from_x0_gridline_pt'] > 0), 'small_support_clear_of_slope1_reference_line': bool(small_support_clearances['from_slope1_reference_line_pt'] > 0), 'small_support_inside_right_axis_boundary': bool(small_support_clearances['to_right_axis_boundary_pt'] > 0), 'small_support_below_group_divider': bool(small_support_clearances['below_y4_group_divider_pt'] > 0), 'small_support_above_first_small_row': bool(small_support_clearances['above_first_small_support_row_pt'] > 0), 'nine_pooled_values_unchanged': bool(association == expected_association), 'three_double_cold_mean_sd_values_unchanged': bool(actual_block == expected_block), 'sixteen_family_slope_points_retained': bool(len(slopes) == 16)}}
        assert all(diagnostics['assertions'].values())
        save_figure(fig, 'Figure_S2')
    return diagnostics
