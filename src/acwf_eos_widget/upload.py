import os
import tempfile
import json
from pathlib import Path

class FileManager:
    """Class to manage file uploads and data loading."""
    
    def __init__(self):
        self.available_files = []
        self.temp_dir = tempfile.mkdtemp()
        self.filename_custom_name_map = {}
        self.filename_custom_name_map_rev = {}

    def process_upload(self, files: list) -> None:
        """Process the uploaded files and save them to the temporary directory."""
        for file_info in files:
            filename = file_info['name'].replace('.json', '')
            fd, file_path = tempfile.mkstemp(dir=self.temp_dir, prefix=f"{filename}_")
            file_path = Path(file_path)
            os.close(fd)
  
            with open(file_path, 'wb') as f:
                f.write(file_info['content'])

            if filename not in self.available_files:
                self.available_files.append(file_path.name)
                self._assign_custom_name(file_path.name, filename)
            else:
                print(f"File {filename} already exists in the available files list.")

    def load_data(self, selected_codes: list, ref_code: str, config: dict) -> dict:
        """Load the selected data files and the reference data file."""
        code_results = {}
        short_labels = {}
        
        if not selected_codes:
            raise ValueError("No files selected!")
        if not ref_code:
            raise ValueError("No reference code selected!")
        
        for filename_custom in selected_codes:
            filename = self.filename_custom_name_map_rev.get(filename_custom, filename_custom)
            
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                short_labels[filename_custom] = filename_custom
                code_results[filename_custom] = data
            else:
                print(f"File '{filename}' not found in the temporary folder!")
        
        config['REFERENCE_CODE_LABEL'] = ref_code

        ref_code = self.filename_custom_name_map_rev.get(ref_code, ref_code)
        with open(os.path.join(self.temp_dir, ref_code), 'r', encoding='utf-8') as f:
            ref_data = json.load(f)

        loaded_data = {
            "code_results": code_results,
            "short_labels": short_labels,
            "reference_short_label": ref_code,
            "compare_plugin_data": ref_data
        }

        return loaded_data


    def _assign_custom_name(self, selected_file: str, custom_name: str):
        """Assign a custom name to the currently selected file."""
        if not selected_file:
            print("No file selected.")
            return
        
        if not custom_name:
            print("Please enter a custom name.")
            return
        
        self.available_files[
            self.available_files.index(
                self.filename_custom_name_map.get(selected_file, selected_file)
                )
            ] = custom_name
        
        self.filename_custom_name_map[selected_file] = custom_name
        self.filename_custom_name_map_rev[custom_name] = self.filename_custom_name_map_rev.get(
            selected_file, selected_file
            )
        # print(f"Assigned custom name '{custom_name}' to '{selected_file}'.")
