"""
This module contains feature names for different feature sets.

Currently, this module is compatible with column name from itteration 5.
Example dataset file: dataset_7June20225_itteration5_CO2only.xlsx.
"""

X_RAW_DATA = ["Eapp (V)", "I (mA)", "R (Ω)", "global_time", "global_Q"]

X_METADATA_NUMERICAL = [
    "cathode_geometric_surface_area",
    "cathode_gas_flow",
    "cathode_set_ph",
    "cathodic_chloride_conc",
    "cathodic_sulphate_conc",
    "cathodic_potassium_ion_conc",
    "cathodic_sodium_ion_conc",
    "cathodic_hydrogencarbonate_ion_conc",
    "cathode_volumne",
    "anode_electrolyte_flow",
    "anode_electrolyte_ph",
    "anode_electrolyte_volume",
    "anode_gas_flow",
    "cathode_catalyst_loading",
    "cathode_electrolyte_flow",
    "cathode_catalyst_thickness",
    "anodic_chloride_conc",
    "anodic_sulphate_conc",
    "anodic_potassium_ion_conc",
    "anodic_hydrogencarbonate_ion_conc",
    "anodic_phosphate_ion_conc",
    "cathode_substrate_pore_size",
]

X_METADATA_CATEGORICAL = [
    "ion_exchange_membrane",
    "cathode_gas_mix",
    "cathode_reaction",
    "electrical_connection_type",
    "cathode_substrate_name",
    "anodic_gas_mixture",
    "cathode_substrate_chemical_formula",
]

X_ENGINEERED = [
    "global_time_positive_i",
    "delta_time_positive_i",
    "global_time_negative_i",
    "delta_time_negative_i",
    "delta_time_posneg_ratio",
    "global_time_posneg_ratio",
    "delta_q_pos_i",
    "global_q_pos_i",
    "delta_q_neg_i",
    "global_q_neg_i",
    "delta_q_posneg_ratio",
    "global_q_posneg_ratio",
    "delta_i_neg_avg",
    "delta_i_pos_avg",
    "delta_i_posneg_avg_ratio",
    "delta_e_pos_avg",
    "delta_e_neg_avg",
    "delta_e_posneg_avg_ratio",
    "delta_r",
    "delta_flow_out",
    "delta_Eapp",
    "delta_Eapp_fluctuation",
]

PRODUCTS = ["fe-H2", "fe-C2H4", "fe-CH4"]

X_ALL_NUMERICAL = X_RAW_DATA + X_ENGINEERED + X_METADATA_NUMERICAL

X_ALL = X_RAW_DATA + X_ENGINEERED + X_METADATA_NUMERICAL + X_METADATA_CATEGORICAL
