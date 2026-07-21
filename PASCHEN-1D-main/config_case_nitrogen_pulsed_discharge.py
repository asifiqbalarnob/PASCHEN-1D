"""Nitrogen nanosecond pulsed-discharge case."""

from config import SimulationConfig as _BaseSimulationConfig


class SimulationConfig(_BaseSimulationConfig):
    """Apply only case-specific values to the canonical configuration."""

    def __init__(self) -> None:
        super().__init__()
        self.run.run_name = "nitrogen_pulsed_discharge"
        self.run.T_total = 150e-9

        self.numerics.Nt = 150_000
        self.numerics.Nx = 1_000
        self.numerics.kt_limiter_theta = 1.01
        self.numerics.use_adaptive_substepping = False
        self.numerics.max_substeps = 64

        self.geometry.L = 0.01
        self.geometry.A = 0.001
        self.geometry.l = 0.00175

        self.plasma_state.gas = "nitrogen"
        self.plasma_state.p_Torr = 60.0
        self.plasma_state.n0 = 1.0e13
        self.ion_transport.positive_ion = "N2+"

        electron_table = "n2_swarm_output_full_EoverN.dat"
        self.local_field_approximation.electron_transport_source = "user_defined_equation"
        self.local_field_approximation.electron_swarm_data_path = electron_table
        self.townsend_coefficient.townsend_alpha_swarm_data_path = electron_table
        self.ionization_frequency_source.ionization_frequency_swarm_data_path = electron_table

        self.waveform.waveform_type = "gaussian"
        self.waveform.V_peak = 20_000.0
        self.waveform.tV_end = 91e-9

        self.circuit.circuit_type = "dielectric_plasma"
        self.circuit.R0 = 0.0
        self.circuit.R_m = 0.0
        self.circuit.C_ext = 0.0

        self.emission.gamma = 0.3
        self.emission.enable_external_emission = False
        self.emission.enable_cathode_external_emission = False
        self.emission.cathode_enable_quantum_pulse_emission = False
        for prefix in ("shared", "anode", "cathode"):
            setattr(self.emission, f"{prefix}_laser_t0", 10e-6)
            setattr(self.emission, f"{prefix}_laser_U_J", 150e-6)
            setattr(self.emission, f"{prefix}_emission_eps_points", 40)

        self.output.save_every = 30
        self.diagnostics.temporal.quantities = (
            "V_app",
            "V_gap",
            "I_discharge",
            "cfl",
            "particle_inventory",
        )
