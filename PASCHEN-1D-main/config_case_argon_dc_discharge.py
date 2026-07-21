"""Argon 140 V DC glow-discharge case."""

from config import SimulationConfig as _BaseSimulationConfig


class SimulationConfig(_BaseSimulationConfig):
    """Apply only glow-discharge values to the canonical configuration."""

    def __init__(self) -> None:
        super().__init__()
        self.run.run_name = "argon_dc_glow_discharge_140V"
        self.run.T_total = 40e-6

        self.numerics.Nt = 4_000_000
        self.numerics.kt_limiter_theta = 1.01
        self.numerics.use_adaptive_substepping = False
        self.numerics.max_substeps = 64

        self.plasma_state.p_Torr = 2.88
        self.plasma_state.n0 = 1.0e14
        self.ion_transport.positive_ion = "Ar+"

        self.local_field_approximation.electron_transport_source = "user_defined_equation"

        self.waveform.V_peak = 140.0
        self.waveform.tV_end = 40e-6

        self.circuit.R0 = 0.0
        self.circuit.C_ext = 0.0

        self.emission.gamma = 0.05
        self.emission.enable_external_emission = False
        self.emission.enable_anode_external_emission = False
        self.emission.enable_cathode_external_emission = False
        self.emission.cathode_enable_quantum_pulse_emission = False
        for prefix in ("shared", "anode", "cathode"):
            setattr(self.emission, f"{prefix}_laser_t0", 10e-6)
            setattr(self.emission, f"{prefix}_laser_U_J", 150e-6)
            setattr(self.emission, f"{prefix}_emission_eps_points", 40)

        self.output.save_every = 800
