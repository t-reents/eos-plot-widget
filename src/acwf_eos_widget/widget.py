from ipywidgets import widgets
from IPython.display import display, clear_output
import warnings

from acwf_eos_widget.upload import FileManager
from acwf_eos_widget.select import UnariesOxidesSelector, Selector
from acwf_eos_widget.utils import (
    run_periodic_tables,
    _get_data_heatmatp_comparison,
    compare_properties,
    filterable_scrollable_df,
    compare_properties_plotly,
    make_quality_matching_cmap
    )
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from pathlib import Path

from acwf_paper_plots.plots.common.generate_periodic_tables import _get_config

# Base class for result widgets
class BaseWidget:
    def __init__(self, file_manager, selector, unaries_oxides_selector):
        self.file_manager = file_manager
        self.selector = selector
        self.unaries_oxides_selector = unaries_oxides_selector
        self.output = widgets.Output()
        self.process_button = widgets.Button(
            description='Process', style={'description_width':'initial'}, layout={'width':'150px'}
        )
        self.process_button.on_click(self._on_process)

    def get_children(self):
        raise NotImplementedError

    def _on_process(self, change):
        raise NotImplementedError

# Widget for periodic table plots
class PeriodicTableWidget(BaseWidget):
    def __init__(self, file_manager, selector, unaries_oxides_selector):
        super().__init__(file_manager, selector, unaries_oxides_selector)
        self.process_button.description = 'Process Selected Files'
        self.process_button.layout.width = '250px'
        self.config = _get_config()
        self.config['USE_AE_AVERAGE_AS_REFERENCE'] = False
        self.config['SHOW_PLOT'] = 'notebook'

    def _on_process(self, change):
        self.unaries_oxides_selector.validate_selection()
        set_name = self.unaries_oxides_selector.set_name
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.output.clear_output()
            with self.output:
                selected_files = self.selector.codes_selector.value
                selected_ref = self.selector.ref_selector.value
                if not selected_files:
                    raise ValueError('No files selected!')
                data = self.file_manager.load_data(selected_files, selected_ref, self.config)
                run_periodic_tables(data, set_name, self.config)
                print('Processing complete.')

    def get_children(self):
        return widgets.VBox([
            widgets.HTML(
                value=(
                    '<h2>Create periodic tables</h2>'
                    '<p>Select multiple codes in <b>Codes to select</b> and the reference in <b>Reference code</b> above. '
                    'The periodic tables show the <b>epsilon</b> and <b>nu</b> metrics across all the configurations '
                    'for each of the selected codes compared to the selected reference.</p>'
                )
                ),
            self.process_button,
            self.output
        ])


class MainWidget:
    def __init__(self):
        self.file_manager = FileManager()
        self.file_uploader = widgets.FileUpload(accept='.json', multiple=True)
        self.selector = Selector(self.file_manager)
        self.unaries_oxides_selector = UnariesOxidesSelector()
        # Sub-widgets
        self.periodic_widget = PeriodicTableWidget(
            file_manager=self.file_manager,
            selector=self.selector,
            unaries_oxides_selector=self.unaries_oxides_selector
        )
        self.heatmap_widget = HeatmapWidget(
            file_manager=self.file_manager,
            selector=self.selector,
            unaries_oxides_selector=self.unaries_oxides_selector
        )
        self.property_widget = PropertyComparisonWidget(
            file_manager=self.file_manager,
            selector=self.selector,
            unaries_oxides_selector=self.unaries_oxides_selector
        )
        # Observers
        self.file_uploader.observe(self._upload_files, names='value')
        
        self._load_reference_data()

    def _load_reference_data(self):
        ref_path = Path(__file__).parent / 'reference-data'
        reference_data = []
        if ref_path.exists():
            for file_path in ref_path.glob('*.json'):
                reference_data.append(
                    {
                        'name': file_path.name,
                        'content': file_path.read_bytes()
                    }
                )
        self.file_manager.process_upload(reference_data)
        self.selector.update_available_fields(self.file_manager.available_files)

    def _upload_files(self, _change):
        if self.file_uploader.value:
            self.file_manager.process_upload(self.file_uploader.value)
            self.selector.update_available_fields(self.file_manager.available_files)

    def display(self):

        tab = widgets.Tab(children=[
            self._periodic_tab(),
            self.heatmap_widget.get_children(),
            self.property_widget.get_children()
        ])
        tab.set_title(0, 'Periodic tables')
        tab.set_title(1, 'Heatmap')
        tab.set_title(2, 'EOS parameters')
        def on_tab_change(change):
            if change['new'] == 1:
                self.heatmap_widget.update_config_options()
        tab.observe(on_tab_change, names='selected_index')


        main_ui = widgets.VBox([
            widgets.HTML(value="""
                <h2>Upload json files</h2>
                <p>Upload all the json files that you want to use <b>(by default, the average of WIEN2k, FLEUR and CP2k+SIRIUS – WiFlCp is provided as reference)</b>.</p>
            """),
            self.file_uploader,
            self.selector,
            widgets.HTML(value="<h2>Select whether you are comparing the unaries or oxides</h2>"),
            self.unaries_oxides_selector,
            widgets.HTML(value="<h2></h2>"),
            tab
        ])
        display(main_ui)

    def _periodic_tab(self):
        return widgets.VBox([
            widgets.HTML(value='<h2>Create periodic tables</h2>'),
            self.periodic_widget.process_button,
            self.periodic_widget.output
        ])

# New widget to display an epsilon heatmap
class HeatmapWidget:
    """Widget that displays a seaborn heatmap of epsilon comparisons with extra controls."""
    def __init__(self, file_manager, selector, unaries_oxides_selector):
        self.file_manager = file_manager
        self.selector = selector
        self.unaries_oxides_selector = unaries_oxides_selector
        self.output = widgets.Output()

        self.chem_element_text = widgets.Text(
            value='',
            description='Element:',
            style={'description_width': 'initial'},
            layout={'width': '150px'}
        )

        self.config_dropdown = widgets.Dropdown(
            options=[],
            description='Configuration:',
            style={'description_width': 'initial'},
            layout={'width': '200px'}
        )
        self.process_button = widgets.Button(
            description="Update Heatmap",
            style={'description_width': 'initial'},
            layout={'width': '150px'}
        )
        self.process_button.on_click(self._update_heatmap)
        self.unaries_oxides_selector.button_unaries.observe(self.update_config_options, names='value')

    def update_config_options(self, change=None):
        configs = []
        if self.unaries_oxides_selector.set_name == 'unaries':
            configs = ['FCC', 'BCC', 'SC', 'DIA']
        elif self.unaries_oxides_selector.set_name == 'oxides':
            configs = [
                'X2O', 'X2O3', 'XO', 'X2O5', 'XO2', 'XO3'
            ]
        
        self.config_dropdown.options = configs
        
    def _load_data(self, selected_codes, sub_keys=None):
        output = {}
        for filename_custom in selected_codes:
            filename = self.file_manager.filename_custom_name_map_rev.get(filename_custom, filename_custom)
            file_path = os.path.join(self.file_manager.temp_dir, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{filename}' not found in the temporary folder!")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if sub_keys is None:
                sub_keys = data.keys()
            if any(sub_key not in data for sub_key in sub_keys):
                raise KeyError(f"One of the sub-keys {sub_keys} not found in the data for file '{filename_custom}'!")
            output[filename_custom] = {
                sub_key: data[sub_key] for sub_key in sub_keys
            }
        return output

    def _update_heatmap(self, _change):
        selected_files = self.selector.codes_selector.value
        data = self._load_data(selected_files, ['BM_fit_data', 'num_atoms_in_sim_cell'])
        ref_data = self._load_data((self.selector.ref_selector.value,), ['BM_fit_data', 'num_atoms_in_sim_cell'])
        data.update(ref_data)
        self._render_heatmap(data)

    def _render_heatmap(self, data):
        unaries_oxides_sep = {
            'unaries': '-X/',
            'oxides': '-'
        }
        self.output.clear_output(wait=True)
        
        norm_nu, cmap_nu = make_quality_matching_cmap("nu")
        norm_eps, cmap_eps = make_quality_matching_cmap("epsilon")
        with self.output:
            configuration = (
                self.chem_element_text.value + unaries_oxides_sep[self.unaries_oxides_selector.set_name] + self.config_dropdown.value
            )
            configuration = configuration.replace('DIA', 'Diamond')
            eps_dict, nu_dict = _get_data_heatmatp_comparison(data, configuration)
            name_mapping = {
                n: i for i, n in enumerate(sorted(eps_dict.keys()), start=1)
            }
            name_mapping_rev = {
                v: k for k, v in name_mapping.items()
            }
            print('\n\nThe indices of the codes are:')
            print('\n'.join([f'\t{i}: {name}' for i, name in name_mapping_rev.items()]))
            
            df_eps = pd.DataFrame(eps_dict)
            df_nu = pd.DataFrame(nu_dict)
            df_eps.rename(index=name_mapping, columns=name_mapping, inplace=True)
            df_nu.rename(index=name_mapping, columns=name_mapping, inplace=True)
            _fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            
            plt.subplots_adjust(wspace=0.3)
            
            sns.heatmap(
                df_eps, annot=True, ax=ax[0], cbar_kws={'label': r'$\epsilon$'}, cmap=cmap_eps, norm=norm_eps
                )
            ax[0].set_title(r'$\epsilon$ Heatmap')
            
            sns.heatmap(df_nu, annot=True, ax=ax[1], cbar_kws={'label': r'$\nu$'}, cmap=cmap_nu, norm=norm_nu)
            ax[1].set_title(r'$\nu$ Heatmap')
            
            plt.show()

    def get_children(self):
        controls = widgets.HBox([self.chem_element_text, self.config_dropdown])
        # Render the heatmap with current control values
        return widgets.VBox(
            [
                widgets.HTML(
                    value=(
                        '<h2>EOS comparison across multiple codes for a specific configuration.</h2>'
                        '<p>Select multiple codes in <b>Codes to select</b> that will be compared against each other (including the potentially selected reference code). '
                        'Next, type the element in the text field and select the configuration of interest in '
                        'the dropdown menu. The configurations will be updated in case you siwtch between unaries and oxides above. </p>' 
                        '<p>The heatmap shows the <b>epsilon</b> and <b>nu</b> of the pairwise EOS comparisons. </p>'
                    )
                ),
                controls, self.process_button, self.output
                ]
            )

class PropertyComparisonWidget(BaseWidget):

    def __init__(self, file_manager, selector, unaries_oxides_selector):
        super().__init__(file_manager, selector, unaries_oxides_selector)
        self.process_button.description = 'Compare Properties'
        self.process_button.layout.width = '200px'
        self.table = None

    def _load_data(self, selected_codes):
        output = {}
        for filename_custom in selected_codes:
            filename = self.file_manager.filename_custom_name_map_rev.get(filename_custom, filename_custom)
            file_path = os.path.join(self.file_manager.temp_dir, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{filename}' not found!")
            with open(file_path, 'r', encoding='utf-8') as f:
                output[filename_custom] = json.load(f)
        return output

    def _on_process(self, _change):
        selected = self.selector.codes_selector.value
        data = self._load_data(selected)
        ref = self._load_data((self.selector.ref_selector.value,))
        ref = ref[self.selector.ref_selector.value]
        
        with self.output:
            clear_output(wait=True)
            table_df, table_widget = compare_properties_plotly(data, ref)
            table_widget.show(renderer='notebook')
            display(filterable_scrollable_df(table_df, column='rel_diff'))

    def get_children(self):
        return widgets.VBox([
            self.process_button,
            widgets.HTML(
                value=(
                    '<h2>Relative differences per code and property with respect to the reference.</h2>'
                    '<p>Select multiple codes in <b>Codes to select</b> and the reference in <b>Reference code</b> above. '
                    'The box plot is interactive, e.g., you can zoom.</p>' 
                    '<p>The (scrollable) table below shows the corresponding data (the code, '
                    'the configuration, the value of the EOS parameter, the value of the reference and the relative difference in %. '
                    'You can sub-select the presented EOS parameters in the multi-select widget and also filter by the abs. '
                    'value of the relative difference, to only include those configurations with abs. relative differences above that threshold. '
                    'This feature might be helpful to identify outliers.</p>'
                )
                ),
            self.output,
        ])