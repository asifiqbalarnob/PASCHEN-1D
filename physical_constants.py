"""
physical_constants.py

Shared physical constants (SI units) for the PASCHEN-1D
drift-diffusion-Poisson plasma code.

These constants are kept in a dedicated module to ensure:
  • Numerical consistency across all physics and circuit modules
  • Easy modification or extension (e.g., adding ion masses,
    collision cross-sections, or gas-dependent parameters)
  • Cleaner code in physics.py, emission.py, circuit.py, etc.

All constants follow CODATA values to reasonable precision for
plasma simulation purposes. If higher precision is ever required,
they can be substituted here without touching the rest of the code.
"""

from math import pi

# ------------------------------------------------------------
# Fundamental physical constants (SI)
# ------------------------------------------------------------

#: Boltzmann constant [J/K]
#:   Relates temperature to energy (k_B T).
kB = 1.3807e-23

#: Vacuum permittivity eps0 [F/m]
#:   Appears in Poisson equation and in dielectric models.
eps0 = 8.85e-12

#: Speed of light in vacuum [m/s]
#:   Used mainly in emission normalization (laser frequency).
c = 3.0e8

#: Elementary charge e [C]
#:   Magnitude of the electron/proton charge. Critical for
#:   charge density ρ = e (n_i – n_e) and flux definitions.
e = 1.602e-19

#: Electron mass mₑ [kg]
#:   Used in quantum emission formulas and ponderomotive energy.
m_e = 9.1093837e-31

#: Reduced Planck constant ħ [J·s]
#:   Needed in the photoemission model (Airy function branch).
hbar = 6.62607015e-34 / (2.0 * pi)


# ------------------------------------------------------------
# Notes
# ------------------------------------------------------------
# • All constants are defined in SI units.
# • Accuracy is sufficient for plasma modeling; if desired,
#   CODATA 2018/2022 constants can be substituted directly.
# • If the code later requires ion masses (Ar⁺, N₂⁺, etc.),
#   they should be added here for consistency.
# Example additions (uncomment when needed):
# m_p = 1.67262192369e-27   # proton mass [kg]
# m_Ar = 6.6335209e-26      # argon ion mass [kg]
# m_N2 = 4.65e-26           # nitrogen ion mass [kg]
# • Keep this module minimal—only constants, no functions.
