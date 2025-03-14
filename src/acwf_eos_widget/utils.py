from acwf_paper_plots.plots.common.generate_periodic_tables2 import (
    calculate_quantities,
    find_code_measures_max_and_avg,
    plot_periodic_tables,
    analyze_stats
)

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