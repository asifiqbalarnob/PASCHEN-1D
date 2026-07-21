"""Helium photoemission smoke-test case."""

from config import SimulationConfig as _BaseSimulationConfig


class SimulationConfig(_BaseSimulationConfig):
    def __init__(self) -> None:
        super().__init__()
        self.run.run_name = "helium_photoemission_smoke_test"
        self.plasma_state.gas = "helium"
        self.plasma.electron_kinetics_model = "local_field_approximation"
        electron_table = "he_swarm_output_full_EoverN.dat"
        self.local_field_approximation.electron_swarm_data_path = electron_table
        self.townsend_coefficient.townsend_alpha_source_mode = (
            "interpolate_from_e_over_n_table"
        )
        self.townsend_coefficient.townsend_alpha_swarm_data_path = electron_table
        self.ionization_frequency_source.ionization_frequency_swarm_data_path = electron_table

        self.plasma.ion_kinetics_model = "local_field_ion_kinetics"
        self.ion_transport.positive_ion = "4He+"
        self.ion_transport.mobility_source_mode = "swarm_data_table_interpolation"
        self.ion_transport.diffusion_source_mode = "einstein_relation"
        self.ion_transport.mobility_table_path = (
            "lxcat_57511c29d0dafef749d3_reduced_mobility.csv"
        )
