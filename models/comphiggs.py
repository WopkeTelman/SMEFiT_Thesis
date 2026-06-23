"""
models/comphiggs.py

Minimal Composite Higgs Model (MCHM) — SO(5)/SO(4) coset.

UV parameters:
    g_rho   : strong sector coupling (naturalness range 1–4π)
    m_rho   : mass of lightest composite resonance [TeV]

Derived scales:
    f       = m_rho / g_rho    Goldstone decay constant [TeV]
    xi      = v² / f²          compositeness parameter  (v = 0.246 TeV)

Matching relations: SILH Lagrangian at leading order in xi,
projected onto Warsaw basis operators, in units of TeV^{-2}.

Operator content (4 operators):

  OpBox  (c_{H□})   = -g_rho² / (2 m_rho²)
    Kinetic term of the non-linear sigma model:
      L ⊃ (1/(2f²)) (∂_μ |H|²)²
    After integration by parts this generates O_{H□} with coefficient
    c_{H□} = -1/(2f²) = -g_rho²/(2 m_rho²).

  Otap   (c_{tH})   = -y_t · g_rho / m_rho²    ← LINEAR in g_rho
    Top partial compositeness (MCHM5 embedding):
    one insertion of the strong-sector Yukawa y_* ∼ g_rho and one SM
    top Yukawa y_t = m_t/v ≈ 0.703 (at m_t = 173 GeV, v = 246 GeV):
      c_{tH} = -(y_* · y_t) / m_rho²,  y_* = g_rho
    The linear dependence on g_rho is what makes the UV Jacobian
    rank-2 (Otap lives on a different curve than the g_rho²/m_rho²
    operators), allowing the 2-UV-parameter fit to close non-trivially.

  O3pQ3  (c_{Hq}^{(3)}) = +g_rho² / (4 m_rho²)
    Tree-level exchange of a composite SU(2)_L triplet vector ρ_μ:
      g_rho ρ_μ^a (q̄_L T^a γ^μ q_L) + g_rho ρ_μ^a (H† T^a D^μ H)
    → contact term ∼ +1/(4f²) after integrating out ρ.

  OpQM   (c_{Hq}^{(-)}) = -g_rho² / (4 m_rho²)
    Parity partner of O3pQ3 from the same vector exchange:
      c_{Hq}^{(-)} = c_{Hq}^{(3)} − c_{Hq}^{(1)} = −1/(4f²).
    (c_{Hq}^{(1)} = +1/(2f²) from the singlet component of ρ;
     c_{Hq}^{(3)} = +1/(4f²) from the triplet component → difference = −1/(4f²).)

Discriminating power against W' and Z':
    W' fingerprint: OQl (quark-lepton 4-fermion) + O3pl (Higgs-lepton).
    Z' fingerprint: OpD (T-parameter) + Opl (Higgs-lepton).
    CHM fingerprint: OpBox (Higgs kinetic) + Otap (ttH) + quark-Higgs
      — all Higgs-sector operators without 4-fermion contact terms.
    The 4-operator set is non-overlapping with the leading W'/Z' operators.

References:
    Giudice, Grojean, Pomarol, Rattazzi (2007): JHEP 06 045  [hep-ph/0703164]
    Contino, Ghezzi, Grojean, Muhlleitner, Spira (2013): JHEP 07 035
    Elias-Miro, Espinosa, Grojean, Masso, Wulzer (2013): JHEP 08 033
"""

from dataclasses import dataclass
from typing import Dict, List

# Top Yukawa coupling: y_t = m_t / v = 173.0 / 246.0 ≈ 0.7033
_Y_TOP: float = 173.0 / 246.0


@dataclass
class CompHiggsModel:
    """
    Minimal Composite Higgs model (MCHM5, SO(5)/SO(4)).

    Parameters
    ----------
    g_rho : strong sector coupling  (typical range 1–4)
    m_rho : resonance mass [TeV]    (typical range 1–20 TeV)
    """
    g_rho: float = 2.0
    m_rho: float = 10.0

    # Warsaw-basis operators generated at the matching scale (class-level, not fields)
    OPERATORS = ["OpBox", "Otap", "O3pQ3", "OpQM"]
    ALL_OPS   = OPERATORS

    def eft_coefficients(self) -> Dict[str, float]:
        """
        EFT Wilson coefficients at the matching scale m_rho [TeV^{-2}].

          OpBox : -(1/2) g_rho² / m_rho²   sigma-model kinetic
          Otap  : -y_t   g_rho  / m_rho²   top partial compositeness (linear in g!)
          O3pQ3 : +(1/4) g_rho² / m_rho²   SU(2)_L vector exchange
          OpQM  : -(1/4) g_rho² / m_rho²   parity partner of O3pQ3
        """
        g  = self.g_rho
        m2 = self.m_rho ** 2
        return {
            "OpBox": -0.5    * g**2 / m2,
            "Otap":  -_Y_TOP * g    / m2,
            "O3pQ3": +0.25   * g**2 / m2,
            "OpQM":  -0.25   * g**2 / m2,
        }

    def uv_coeff_block(self, prior_scale: float = 5.0) -> Dict:
        """
        smefit UV coupling COEFF_BLOCK for Nested Sampling runcard.

        Encoding convention (same as WPrimeModel):
          c = prefactor × param^power  → [prefactor, power] pair per parameter.

        Note: Otap has power_g = 1 (linear) while the other three have power_g = 2.
        m_rho_TeV is fixed (constrained) at self.m_rho; g_rho is the free UV parameter.
        """
        prior_max = prior_scale * self.g_rho
        return {
            "OpBox":     {"constrain": [{"g_rho": [-0.5,    2], "m_rho_TeV": [1.0, -2]}]},
            "Otap":      {"constrain": [{"g_rho": [-_Y_TOP, 1], "m_rho_TeV": [1.0, -2]}]},
            "O3pQ3":     {"constrain": [{"g_rho": [ 0.25,   2], "m_rho_TeV": [1.0, -2]}]},
            "OpQM":      {"constrain": [{"g_rho": [-0.25,   2], "m_rho_TeV": [1.0, -2]}]},
            "g_rho":     {"min": 0.0, "max": float(prior_max)},
            "m_rho_TeV": {"constrain": True, "value": float(self.m_rho)},
        }

    def uv_param_names(self) -> List[str]:
        return ["g_rho"]   # m_rho is fixed (constrain: True in uv_coeff_block)

    def uv_truth(self) -> Dict[str, float]:
        return {"g_rho": self.g_rho}   # mass excluded — mirrors WPrimeModel convention

    @property
    def f(self) -> float:
        """Goldstone decay constant f = m_rho / g_rho [TeV]."""
        return self.m_rho / self.g_rho

    @property
    def xi(self) -> float:
        """Compositeness parameter xi = v² / f² (v = 0.246 TeV)."""
        return (0.246 / self.f) ** 2

    def __repr__(self) -> str:
        return (
            f"CompHiggsModel(g_rho={self.g_rho}, m_rho={self.m_rho} TeV, "
            f"f={self.f:.3f} TeV, xi={self.xi:.4f}) [4 ops]"
        )
