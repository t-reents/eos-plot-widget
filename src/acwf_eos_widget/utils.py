from acwf_paper_plots.plots.common.generate_periodic_tables import (
    calculate_quantities,
    find_code_measures_max_and_avg,
    plot_periodic_tables,
    analyze_stats
)
import ipywidgets as widgets
from IPython.display import display
from itertools import product
import acwf_paper_plots.quantities_for_comparison as qc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from bokeh.plotting import figure, show
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import jitter
import numpy as np
from bokeh.io import output_notebook
from matplotlib.colors import Normalize, LinearSegmentedColormap



def make_quality_matching_cmap(quantity):
    """
    Create a matplotlib colormap and normalization that matches the
    excellent/good/outlier thresholds for a given quantity.
    """
    # Get thresholds for the quantity
    NU_EPS_FACTOR = 1.65

    EXCELLENT_AGREEMENT_THRESHOLD = {
        'nu': 0.10, 'epsilon': 0.06,
        'delta_per_formula_unit': 0.,  # not used in this script
        'delta_per_formula_unit_over_b0': 0.  # not used in this script
        }
    GOOD_AGREEMENT_THRESHOLD = {
        'nu': 0.33, 'epsilon': 0.20,
        'delta_per_formula_unit': 0.,  # not used in this script
        'delta_per_formula_unit_over_b0': 0.  # not used in this script
        }
    OUTLIER_THRESHOLD = {
        'nu': 1.0 * NU_EPS_FACTOR, 'epsilon': 1.0,
        'delta_per_formula_unit': 0.,  # not used in this script
        'delta_per_formula_unit_over_b0': 0.  # not used in this script
        }

    # Optional: if you have a custom maximum value for the colorbar per quantity,
    # define CBAR_MAX_DICT. Otherwise, you can leave it empty.
    CBAR_MAX_DICT = {}  # e.g., {'nu': 0.2, 'epsilon': 0.1}
        
    exc_thresh = EXCELLENT_AGREEMENT_THRESHOLD[quantity]
    good_thresh = GOOD_AGREEMENT_THRESHOLD[quantity]
    outl_thresh = OUTLIER_THRESHOLD[quantity]
    
    # Set the maximum value of the colorbar (a slight extension beyond the outlier value)
    colorbar_max = 1.04 * outl_thresh
    
    # Define breakpoints and corresponding colors
    cvals  = [0.0, exc_thresh, good_thresh, outl_thresh, outl_thresh+0.001, colorbar_max]
    colors = ["#555998", "#6B71AD", "#EEE992", "#f53216", "#bf0000", "#bf0000"]
    
    # Create a Normalize object based on the defined values
    norm = Normalize(vmin=min(cvals), vmax=max(cvals))
    
    # Map the thresholds to positions between 0 and 1
    positions = [norm(val) for val in cvals]
    
    # Build a linear segmented colormap from the given breakpoints and colors
    cmap = LinearSegmentedColormap.from_list("quality_matching", list(zip(positions, colors)), N=256)
    
    # If a maximum value is specified for the colorbar for this quantity, update the normalization
    if quantity in CBAR_MAX_DICT:
        high = CBAR_MAX_DICT[quantity]
        norm = Normalize(vmin=min(cvals), vmax=high)
    
    return norm, cmap

def run_periodic_tables(data: dict, set_name: str, config: dict) -> None:

    master_data_dict = {}
    config['SET_NAMES'] = [set_name]
    
    master_data_dict[set_name] = {
        "loaded_data": data,
        "calculated_quantities": {}
    }
    for QUANTITY in config['QUANTITIES']:
        master_data_dict[set_name]["calculated_quantities"][QUANTITY] = {}
        for plugin, plugin_data in data["code_results"].items():
            collect = calculate_quantities(plugin_data, data["compare_plugin_data"], QUANTITY)
            master_data_dict[set_name]["calculated_quantities"][QUANTITY][plugin] = collect

    measures_max_and_avg = find_code_measures_max_and_avg(master_data_dict, config)

    for QUANTITY in config['QUANTITIES']:
        plot_periodic_tables(set_name, QUANTITY, measures_max_and_avg, master_data_dict, config)

    analyze_stats(master_data_dict, config)
    

def _get_data_heatmatp_comparison(data: dict, configuration: str):

    DEFAULT_wb0 = 1.0/20.0
    DEFAULT_wb1 = 1.0/400.0

    eos_compare_eps = {}
    eos_compare_nu = {}

    eos_mapping = {
        k: v[configuration] for k, v in data.items()
    }
    for comb in product(eos_mapping.keys(), eos_mapping.keys()):
        epsilon = qc.epsilon(
            eos_mapping[comb[0]]['min_volume'], eos_mapping[comb[0]]['bulk_modulus_ev_ang3'], eos_mapping[comb[0]]['bulk_deriv'],
            eos_mapping[comb[1]]['min_volume'], eos_mapping[comb[1]]['bulk_modulus_ev_ang3'], eos_mapping[comb[1]]['bulk_deriv'],
            1, DEFAULT_wb0, DEFAULT_wb1
        )
        
        nu = qc.nu(
            eos_mapping[comb[0]]['min_volume'], eos_mapping[comb[0]]['bulk_modulus_ev_ang3'], eos_mapping[comb[0]]['bulk_deriv'],
            eos_mapping[comb[1]]['min_volume'], eos_mapping[comb[1]]['bulk_modulus_ev_ang3'], eos_mapping[comb[1]]['bulk_deriv'],
            100, DEFAULT_wb0, DEFAULT_wb1
        )

        eos_compare_eps.setdefault(comb[0], {})[comb[1]] = epsilon
        eos_compare_nu.setdefault(comb[0], {})[comb[1]] = nu

    return eos_compare_eps, eos_compare_nu


def scrollable_df(df, height='300px', width='100%'):
    """
    Wraps a pandas DataFrame in an ipywidget.HTML with CSS to make it scrollable.
    
    Parameters:
    - df: pandas.DataFrame to display.
    - height: CSS height of the scrollable container (e.g., '300px', '50vh').
    - width: CSS width of the scrollable container (e.g., '100%', '500px').
    
    Returns:
    - An ipywidgets.HTML widget that you can display in a Jupyter notebook.
    """
    # Convert DataFrame to HTML table (without inline styles)
    html_table = df.to_html(classes='scrollable-table', index=False)
    
    # Define CSS for scrolling and table styling
    css = f"""
    <style>
      .scrollable-table-container {{
        max-height: {height};
        width: {width};
        overflow: auto;
        border: 1px solid lightgray;
      }}
      .scrollable-table {{
        border-collapse: collapse;
        width: 100%;
      }}
      .scrollable-table th, .scrollable-table td {{
        border: 1px solid lightgray;
        padding: 4px;
        text-align: left;
      }}
      .scrollable-table th {{
        background-color: #f2f2f2;
        position: sticky;
        top: 0;
        z-index: 1;
      }}
    </style>
    """
    
    widget = widgets.HTML(value=css + f'<div class="scrollable-table-container">{html_table}</div>')
    return widget


def filterable_scrollable_df(df, height='300px', width='100%', column=None):
    """
    Wrap a DataFrame in a slider-controlled scrollable display where rows with values above the slider threshold are hidden.

    Parameters:
    - df: pandas.DataFrame to display.
    - height, width: CSS dimensions for the container.
    - column: optional column name to apply the threshold; if None, filter across all numeric columns.

    Returns:
    - A VBox containing a FloatSlider and the filtered scrollable HTML table.
    """
    column = [column] if column else df.columns
    numeric = df[column].select_dtypes(include=['number']).abs()

    if numeric.empty:
        return widgets.VBox([scrollable_df(df, height, width)])

    min_val = float(numeric.min().min())
    max_val = float(numeric.max().max())
    step = (max_val - min_val) / 100 if max_val != min_val else 1.0
    slider = widgets.FloatSlider(
        value=(min_val + max_val) / 2,
        min=min_val,
        max=max_val,
        step=step,
        description='Abs. relative difference $\ge$:',
        continuous_update=False,
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )
    
    dropdown = widgets.SelectMultiple(
        options=list(df['Property'].unique()),
        description='EOS parameter(s):',
        value=tuple(df['Property'].unique()),
        layout=widgets.Layout(width='40%'),
        style={'description_width': 'initial'}
    )

    out = widgets.Output()
    def update_view(change=None):
        # Filter based on dropdown selection
        selected_props = dropdown.value
        if not isinstance(selected_props, (list, tuple)):
            selected_props = [selected_props]
        if selected_props:
            filtered = df.copy().query("Property in @selected_props")
        else:
            filtered = df.copy()

        # Determine numeric range from the filtered dataframe
        numeric = filtered[column].select_dtypes(include=['number']).abs()
        if not numeric.empty:
            min_val = float(numeric.min().min())
            max_val = float(numeric.max().max())
            slider.min = min_val
            slider.max = max_val
            slider.step = (max_val - min_val) / 100 if max_val != min_val else 1.0
            slider.value = min(slider.value, max_val)

        # Apply slider filtering: if a specific column is provided, use it;
        # otherwise, require all numeric columns to be above the threshold.
        thresh = slider.value
        mask = (numeric >= thresh).all(axis=1)
        final = filtered[mask]

        # Update the output widget
        out.clear_output()
        with out:
            display(scrollable_df(final.rename({
                'config': 'configuration', 
                'Property': 'EOS parameter',
                'Value': 'Code value',
                'avg': 'Reference Value',
                'rel_diff': 'Relative difference (%)',
                }, axis=1), height, width))

    dropdown.observe(update_view, names='value')
    slider.observe(update_view, names='value')
    update_view({'new': slider.value})

    return widgets.VBox([widgets.HBox([dropdown, slider]), out])

def compare_properties(data: dict) -> None:
    # Define the properties to plot. Here we focus on a few key properties from BM_fit_data.
    props = ['min_volume', 'bulk_modulus_ev_ang3', 'bulk_deriv']

    # Create a list of all configurations available in the data
    configs = set()
    for code_data in data.values():
        configs.update(code_data['BM_fit_data'].keys())

    # Prepare the data for plotting
    plot_data = []
    for prop in props:
        for code, code_data in data.items():
            for config in configs:
                if config in code_data['BM_fit_data']:
                    code_config = code_data['BM_fit_data'].get(config, {})
                    if code_config is None:
                        continue
                    value = code_config.get(prop, None)
                    
                    plot_data.append({
                        'config': config,
                        # 'Functional': func.upper(),
                        'Property': prop,
                        'Code': code,
                        'Value': value
                    })

    plot_df = pd.DataFrame(plot_data)

    plot_df['avg'] = plot_df.groupby(
        ['Property', 'config']
        )[['Value']].transform('mean')
    plot_df['rel_diff'] = (
        plot_df['Value'] - plot_df['avg']
        ) / plot_df['avg'] * 100
    # plot_df['rel_diff2'] = (
    #     plot_df['Value'] - plot_df['avg2']
    #     ) / plot_df['avg2'] * 100

    unique_properties = plot_df['Property'].unique()
    n_properties = len(unique_properties)

    xlims = [
        (-0.5, 0.5),
        (-1, 2),
        (-5, 5)
    ]

    xlims = [
        (-7.5, 7.5),
        (-20, 20),
        (-50, 50)
    ]

    pretty_labels = {
        'min_volume': 'Relative difference: $V_0$ (%)',
        'bulk_modulus_ev_ang3': 'Relative difference: $B_0$ (%)',
        'bulk_deriv': "Relative difference: $B_0'$ (%)"
    }

    fig, axs = plt.subplots(
        1, n_properties, figsize=(5 * n_properties, 5),
        sharey=True
        )
    plt.subplots_adjust(wspace=0.3)
    # for i, func in enumerate(unique_functionals):
    for i, prop in enumerate(unique_properties):
        ax = axs[i]
        subset = plot_df[(plot_df['Property'] == prop)]
        # sns.boxplot(data=subset, x='Code', y='Value', ax=ax)
        sns.boxplot(data=subset, y='Code', x='rel_diff', ax=ax)
        # ax.set_title(f"{prop}")
        ax.set_xlabel(pretty_labels[prop])
        ax.set_xlim(xlims[i])
        ax.set_ylabel('')

    # plt.tight_layout()
    plt.show()
    
    return plot_df

def compare_properties_plotly(data: dict, ref: dict) -> tuple:
    """
    Interactive version of compare_properties using Plotly for interactive box plots of relative differences.
    Returns the plotly Figure and DataFrame.
    """
    props = ['min_volume', 'bulk_modulus_ev_ang3', 'bulk_deriv']
    configs = set()
    for code_data in data.values():
        configs.update(code_data['BM_fit_data'].keys())

    plot_data = []
    ref_data = []
    for prop in props:
        for code, code_data in data.items():
            for config in configs:
                code_config = code_data['BM_fit_data'].get(config)
                if not code_config:
                    continue
                value = code_config.get(prop)
                plot_data.append({
                    'config': config,
                    'Property': prop,
                    'Code': code,
                    'Value': value
                })
        for config in configs:
            ref_config = ref['BM_fit_data'].get(config)
            if not ref_config:
                continue
            value = ref_config.get(prop)
            ref_data.append({
                'config': config,
                'Property': prop,
                'avg': value
            })

    ref_df = pd.DataFrame(ref_data)
    plot_df = pd.DataFrame(plot_data)
    # plot_df['avg'] = plot_df.groupby(['Property', 'config'])['Value'].transform('mean')
    plot_df = plot_df.merge(
        ref_df, on=['Property', 'config'])
    
    plot_df['rel_diff'] = (plot_df['Value'] - plot_df['avg']) / plot_df['avg'] * 100

    labels = {
        'rel_diff': 'Relative difference (%)',
        'Code': 'Code'
    }
    
    pretty_labels = {
        'min_volume': 'V<sub>0</sub>',
        'bulk_modulus_ev_ang3': 'B<sub>0</sub>',
        'bulk_deriv': "B<sub>0</sub>'"
    }
    
    fig = px.box(
        plot_df.replace(
            {'Property': pretty_labels}),
        x='rel_diff',
        y='Code',
        color='Code',
        facet_col='Property',
        orientation='h',
        labels=labels,
        category_orders={'Property': list(pretty_labels.values())},
    )

    fig.update_layout(
        # title='Interactive relative differences per code and property with respect to the average.',
        height=500,
        width=500 * len(props),
        showlegend=False,
        hovermode=False
    )
    fig.update_xaxes(matches=None)

    return plot_df, fig