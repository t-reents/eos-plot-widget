from ipywidgets import widgets
from acwf_eos_widget.upload import FileManager

class UnariesOxidesSelector(widgets.HBox):
    """Widget to select whether to plot unaries or oxides."""
    
    def __init__(self):
        self.button_unaries = widgets.ToggleButton(
            description='Unaries',
            value=False,
            layout=widgets.Layout(width='150px'),
            button_style='info'
        )
        self.button_oxides = widgets.ToggleButton(
            description='Oxides',
            value=False,
            layout=widgets.Layout(width='150px'),
            button_style='info'
        )
        self.button_unaries.observe(self.on_button_a_change, names='value')
        self.button_oxides.observe(self.on_button_b_change, names='value')
        
        super().__init__([self.button_unaries, self.button_oxides])
        
    @property
    def set_name(self):
        self.validate_selection()
        return 'unaries' if self.button_unaries.value else 'oxides'
    
    def on_button_a_change(self, change) -> None:
        if change['name'] == 'value':
            if change['new']:
                self.button_oxides.value = False
            else:
                if not self.button_oxides.value:
                    self.button_oxides.value = True
    
    def on_button_b_change(self, change) -> None:
        if change['name'] == 'value':
            if change['new']:
                self.button_unaries.value = False
            else:
                if not self.button_unaries.value:
                    self.button_unaries.value = True

    def validate_selection(self) -> None:
        if not (self.button_unaries.value or self.button_oxides.value):
            raise ValueError("Indicate whether you want to plot unaries or oxides.")


class Selector(widgets.VBox):
    """Widget to select the files to be compared and the reference code."""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager

        self.codes_selector = widgets.SelectMultiple(
            description='Codes to select:',
            style={'description_width': 'initial'},
            disabled=False,
            layout={'width': '450px'}
        )

        self.ref_selector = widgets.Select(
            description='Reference code:',
            style={'description_width': 'initial'},
            disabled=False,
            layout={'width': '450px'}
        )

        self.file_selector = widgets.Select(
            options=[],
            description='Assign new name:',
            style={'description_width': 'initial'},
            rows=6,
            layout={'width': '450px'}
        )
        
        self.custom_name_input = widgets.Text(
            description='Custom Name:',
            style={'description_width': 'initial'},
            layout={'width': '450px', 'margin': '35px 0 0 30px'}
        )
        self.assign_button = widgets.Button(description='Assign Name', style={'description_width': 'initial'})
        
        self.assign_button.on_click(self.assign_custom_name)
        
        super().__init__(
            [
                widgets.HTML(
                    value="""<h2>(optionally) Assign short labels to the codes</h2>
                    <p>Assign custom labels, those will be displayed in the plots (the filenames will be used by default).</p>
                    """
                ),
                widgets.HBox([self.file_selector, self.custom_name_input]),
                self.assign_button,
                widgets.HTML(
                    value="""
                    <h2>Select codes and reference to be compared</h2>
                    <p>Select one or multiple codes to be compared to the reference.</p>
                    """),
                self.codes_selector, 
                widgets.HTML(value="<p>Select a single code that will be used as the reference for all the comparisons.</p>"),
                self.ref_selector, 
                ]
            )

    def update_available_fields(self, available_files: list) -> None:
        """Update the available codes in the selectors"""
        for selector in [
            self.codes_selector, self.file_selector, self.ref_selector
            ]:
            selector.options = available_files
    
    def assign_custom_name(self, change) -> None:
        """Assign a custom name to the selected file"""
        self.file_manager._assign_custom_name(
            self.file_selector.value, self.custom_name_input.value
        )
        self.update_available_fields(self.file_manager.available_files)
        self.custom_name_input.value = ''
