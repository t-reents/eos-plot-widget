from ipywidgets import widgets
from IPython.display import display
import warnings

from acwf_eos_widget.upload import FileManager
from acwf_eos_widget.select import UnariesOxidesSelector, Selector
from acwf_eos_widget.utils import run_periodic_tables

from acwf_paper_plots.plots.common.generate_periodic_tables import _get_config

class Widget:
    """Widget to plot periodic tables comparing EOS of different DFT codes."""

    def __init__(self):
        super().__init__()
        
        self.file_manager = FileManager()
        self.file_uploader = widgets.FileUpload(accept='.json', multiple=True)

        self.selector = Selector(self.file_manager)
        
        self.unaries_oxides_selector = UnariesOxidesSelector()
        
        self.output_area = widgets.Output()
        self.process_button = widgets.Button(
        description="Process Selected Files", style={'description_width': 'initial'},
        layout={'width': '250px'}
        )
        
        self.file_uploader.observe(self.upload_files, names='value')
        self.process_button.on_click(self.plot_periodic_tables)
        
        self.config = _get_config()
        self.config['USE_AE_AVERAGE_AS_REFERENCE'] = False
        self.config['SHOW_PLOT'] = 'notebook'

    def display(self) -> None:
        """Display the widget."""
        widget = widgets.VBox(
            [
                widgets.HTML(
                    value="""
                    <h2>Upload json files</h2>
                    <p>Upload all the json files that you want to use.</p>
                    """),
                self.file_uploader,
                self.selector,
                widgets.HTML(value="<h2>Select whether you are comparing the unaries or oxides</h2>"),
                self.unaries_oxides_selector,
                widgets.HTML(value="<h2>Create periodic tables</h2>"),
                self.process_button,
                self.output_area
            ]
        )
        display(widget)
        

    def upload_files(self, change) -> None:
        """Process the uploaded files."""
        if self.file_uploader.value:
            self.file_manager.process_upload(self.file_uploader.value)
        
            self.selector.update_available_fields(self.file_manager.available_files)

    def plot_periodic_tables(self, change) -> None:
        """Plot the periodic tables."""
        self.unaries_oxides_selector.validate_selection()
    
        set_name = self.unaries_oxides_selector.set_name
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.output_area.clear_output()
            with self.output_area:
                selected_files = self.selector.codes_selector.value
                selected_ref = self.selector.ref_selector.value

                if not selected_files:
                    # print("No files selected!")
                    raise ValueError("No files selected!")
                else:
                    data = self.file_manager.load_data(selected_files, selected_ref, self.config)

                run_periodic_tables(data, set_name, self.config)
                
                print("Processing complete.")

