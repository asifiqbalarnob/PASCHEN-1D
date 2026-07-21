"""Deuterium version of the nitrogen nanosecond pulsed-discharge case."""

from config_case_nitrogen_pulsed_discharge import (
    SimulationConfig as _NitrogenPulsedSimulationConfig,
)


class SimulationConfig(_NitrogenPulsedSimulationConfig):
    """Use BOLSIG+ electron tables and a matched LXCat D3+/D2 ion pair."""

    def __init__(self) -> None:
        super().__init__()
        self.run.run_name = "deuterium_pulsed_discharge"

        self.plasma_state.gas = "deuterium"

        electron_table = "d2_swarm_output_full_EoverN.dat"
        self.plasma.electron_kinetics_model = "local_field_approximation"
        self.local_field_approximation.electron_transport_source = (
            "swarm_data_table_interpolation"
        )
        self.local_field_approximation.electron_swarm_data_path = electron_table
        self.townsend_coefficient.townsend_alpha_source_mode = (
            "interpolate_from_e_over_n_table"
        )
        self.townsend_coefficient.townsend_alpha_swarm_data_path = electron_table
        self.ionization_frequency_source.ionization_frequency_source_mode = (
            "interpolate_from_e_over_n_table"
        )
        self.ionization_frequency_source.ionization_frequency_swarm_data_path = (
            electron_table
        )

        self.plasma.ion_kinetics_model = "local_field_ion_kinetics"
        self.ion_transport.positive_ion = "D3+"
        self.ion_transport.mobility_source_mode = "swarm_data_table_interpolation"
        self.ion_transport.diffusion_source_mode = "swarm_data_table_interpolation"
        self.ion_transport.mobility_table_path = (
            "lxcat_606acd7d3f0d4373cf14_reduced_mobility.csv"
        )
        self.ion_transport.diffusion_table_path = (
            "lxcat_72c9a7cbf9c0f10b3463_reduced_longitudinal_diffusion.csv"
        )
        self.ion_transport.out_of_range_policy = "clip"
        self.ion_transport.gas_temperature_tolerance_K = 1.0
