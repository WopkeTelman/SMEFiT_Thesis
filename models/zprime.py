"""
models/zprime.py

Z' boson model definition for the BSM discovery pipeline.

Model: B'-like (hypercharge-singlet) Z' with universal lepton coupling.

Lagrangian:
    L ⊃ -½ mZp² Z'μ Z'μ
        + gZH  Z'μ (H† i D^μ H) + h.c.    [Higgs current]
        + gZl  Z'μ Σ_gen (l̄_L,gen γ^μ l_L,gen)  [universal left-handed lepton]

Key: This is NOT a sequential Z' (not same quantum numbers as SM Z).
     It is a SU(2)_L×U(1)_Y singlet, coupling only to the H current
     and left-handed lepton current → generates O_{Hl}^(1) not O_{Hl}^(3).

Tree-level matching (Warsaw basis, integrating out Z' at scale mZp):

  Wilson coefficient formula: C_i = prefactor × g_a^p × g_b^q × mZp^(-2)

  1. OpD  (O_{HD} = |H† D_μ H|²):
        C_HD = -gZH² / (2 mZp²)
        Physical origin: Z' propagator × (Higgs current)²
        Factor of 1/2 from integrating out massive vector: L_eff = -1/(2m²)(gJ)²
        sign: negative (shifts T-parameter)

  2. Opl1,2,3  (O_{Hl}^(1) = (H† i↔D_μ H)(l̄_gen γ^μ l_gen)):
        C_{Hl,gen}^(1) = - gZH * gZl / mZp²
        Physical origin: L_eff cross term = -gZH*gZl/mZp² (H†i↔D_μH)(l̄γ^μl)
        sign: negative (from EOM integration, same derivation as C_HD)

  3. Oll_iiii  (O_{ll}^{iiii} = (l̄_i γ^μ l_i)(l̄_i γ_μ l_i), i=1,2,3):
        C_{ll}^{iiii} = - gZl² / (2 mZp²)
        Physical origin: same-generation diagonal term in (Σ_i l̄_i γ^μ l_i)²
        Factor of 1/2 from the overall -1/(2m²) in L_eff

  4. Oll1221  (O_{ll}^{1221} = (l̄_1 γ^μ l_2)(l̄_2 γ_μ l_1)):
        C_{ll}^{1221} = + gZl² / mZp²
        Physical origin: EOM gives -gZl²/m² · O^{1122}; Fierz O^{1122}=-O^{1221}
        → coefficient in SMEFiT basis (ijji form): +gZl²/mZp²

  5. Oll1331  (O_{ll}^{1331} = (l̄_1 γ^μ l_3)(l̄_3 γ_μ l_1)):
        C_{ll}^{1331} = + gZl² / mZp²         [same structure as 1221]

  6. Oll2332  (O_{ll}^{2332} = (l̄_2 γ^μ l_3)(l̄_3 γ_μ l_2)):
        C_{ll}^{2332} = + gZl² / mZp²         [same structure as 1221/1331, gen2×gen3]
        Physical origin: cross term 2×(l̄_2γl_2)(l̄_3γl_3) in (Σ l̄_i γ l_i)²,
        Fierz: O^{2233} = -O^{2332} → C^{2332} = +gZl²/mZp²

  Note: Oll1122, Oll1133, Oll2233 (iijj form) do NOT appear — they are
        eliminated via the Fierz identity in favour of the ijji forms above.

Distinct from W':
  W' generates: O3pl (SU(2) triplet), OpBox, OQl (charged current)
  Z' generates: OpD (T-parameter), Opl (SU(2) singlet), Oll (neutral 4-lepton)
  → completely different Warsaw-basis footprint

All operators confirmed available in smefit FCC-ee theory database.
See: smefit_database/theory/FCCee_*.json, key 'LO'.

Reference: de Blas et al., arXiv:1706.03171 (SMEFT matching dictionary)
           Langacker, arXiv:0801.1345 (Z' review)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class ZPrimeModel:
    """
    B'-like Z' boson UV model (SU(2)_L-singlet neutral vector boson).

    Parameters
    ----------
    gZH : float
        Z'-Higgs current coupling  (H† i D_μ H) Z'^μ
    gZl : float
        Universal Z'-lepton coupling (l̄_L γ_μ l_L) Z'^μ, all generations
    mZp : float
        Z' mass in TeV

    Generates operators: OpD, Opl1, Opl2, Opl3,
                         Oll1111, Oll2222, Oll3333,
                         Oll1221, Oll1331, Oll2332
    All confirmed available in smefit FCC-ee theory database.
    """
    gZH: float = 0.12
    gZl: float = 0.04
    mZp: float = 1.0   # TeV

    # Warsaw basis operators generated at tree-level by this Z' model
    # All confirmed in smefit FCC-ee theory database (LO key)
    OPERATORS = [
        "OpD",      # O_{HD}: T-parameter shift, from (H† D H)² via Z' exchange
        "Opl1",     # O_{Hl,1}^(1): Z'-Higgs-lepton_gen1, singlet H current
        "Opl2",     # O_{Hl,2}^(1): Z'-Higgs-lepton_gen2
        "Opl3",     # O_{Hl,3}^(1): Z'-Higgs-lepton_gen3
        "Oll1111",  # O_{ll}^{1111}: diagonal same-gen term gen1
        "Oll2222",  # O_{ll}^{2222}: diagonal same-gen term gen2
        "Oll3333",  # O_{ll}^{3333}: diagonal same-gen term gen3
        "Oll1221",  # O_{ll}^{1221}: +gZl²/m² after Fierz from direct Z' exchange
        "Oll1331",  # O_{ll}^{1331}: +gZl²/m² (same, gen1×gen3)
        "Oll2332",  # O_{ll}^{2332}: +gZl²/m² (same, gen2×gen3)
    ]

    # smefit operator names (same list — all confirmed in FCC-ee DB)
    # Oll1111/2222/3333 confirmed present: W' model uses them
    SMEFIT_OPERATORS = OPERATORS

    def eft_coefficients(self) -> Dict[str, float]:
        """
        Tree-level EFT Wilson coefficients at matching scale mZp.
        Units: 1/mZp^2 [TeV^-2].

        All signs fixed by Warsaw basis definition with metric (+,-,-,-).
        """
        g  = self.gZH
        gl = self.gZl
        m2 = self.mZp ** 2

        return {
            # T-parameter operator: from (Higgs current)² via Z' propagator
            # Factor 1/2 from L_eff = -1/(2m²)(gJ)² when integrating out Z'
            "OpD":     -g**2  / (2 * m2),

            # Z'-Higgs-lepton vertex (singlet H current × lepton current)
            # Sign: L_eff cross term = -gZH*gZl/mZp² (H†i↔D_μH)(l̄γ^μl)
            "Opl1":    -g * gl / m2,
            "Opl2":    -g * gl / m2,
            "Opl3":    -g * gl / m2,

            # Four-lepton diagonal: same-generation terms from (Σ l̄_i γ l_i)²
            # Coefficient -gZl²/(2m²): factor 1/2 from overall L_eff prefactor
            "Oll1111": -0.5 * gl**2 / m2,
            "Oll2222": -0.5 * gl**2 / m2,
            "Oll3333": -0.5 * gl**2 / m2,

            # Four-lepton off-diagonal: EOM gives -gZl²/m² · O^{iijj},
            # Fierz O^{iijj} = -O^{ijji} → C^{ijji} = +gZl²/m²
            "Oll1221": +gl**2 / m2,
            "Oll1331": +gl**2 / m2,
            "Oll2332": +gl**2 / m2,
        }

    def uv_coeff_block(self, prior_scale: float = 5.0) -> Dict:
        """
        Return smefit UV COEFF_BLOCK for NS runcard.

        Format per operator:
            {"constrain": [{"param": [prefactor, power], ...}]}
        where C_op = prefactor × param^power × ... (product over entries)

        Free UV parameters: gZH, gZl  (mZp is fixed)
        """
        prior_max = min(prior_scale * self.gZH, 2.0)

        return {
            # ── EFT operators constrained by tree-level matching ─────────────

            # OpD = -gZH² / (2 mZp²)
            "OpD":     {"constrain": [{"gZH": [-0.5, 2], "mZp_TeV": [1.0, -2]}]},

            # Opl_gen = -gZH × gZl × mZp^{-2}
            "Opl1":    {"constrain": [{"gZH": [-1.0, 1], "gZl": [1.0, 1], "mZp_TeV": [1.0, -2]}]},
            "Opl2":    {"constrain": [{"gZH": [-1.0, 1], "gZl": [1.0, 1], "mZp_TeV": [1.0, -2]}]},
            "Opl3":    {"constrain": [{"gZH": [-1.0, 1], "gZl": [1.0, 1], "mZp_TeV": [1.0, -2]}]},

            # Oll (diagonal) = -0.5 × gZl² × mZp^{-2}
            "Oll1111": {"constrain": [{"gZl": [-0.5, 2], "mZp_TeV": [1.0, -2]}]},
            "Oll2222": {"constrain": [{"gZl": [-0.5, 2], "mZp_TeV": [1.0, -2]}]},
            "Oll3333": {"constrain": [{"gZl": [-0.5, 2], "mZp_TeV": [1.0, -2]}]},

            # Oll (off-diagonal, ijji form) = +gZl² × mZp^{-2}
            "Oll1221": {"constrain": [{"gZl": [1.0, 2], "mZp_TeV": [1.0, -2]}]},
            "Oll1331": {"constrain": [{"gZl": [1.0, 2], "mZp_TeV": [1.0, -2]}]},
            "Oll2332": {"constrain": [{"gZl": [1.0, 2], "mZp_TeV": [1.0, -2]}]},

            # ── Free UV parameters ────────────────────────────────────────────
            "gZH":     {"min": 0.0, "max": float(prior_max)},
            "gZl":     {"min": 0.0, "max": float(prior_max)},

            # mZp fixed at injection value (not a free parameter in fit)
            "mZp_TeV": {"constrain": True, "value": float(self.mZp)},
        }

    def uv_param_names(self):
        """Names of free UV parameters in the NS fit."""
        return ["gZH", "gZl"]

    def uv_truth(self) -> Dict[str, float]:
        """Injected UV parameter values for closure test."""
        return {"gZH": self.gZH, "gZl": self.gZl}

    def signal_size(self) -> float:
        """
        Characteristic signal size: max |C_i| in TeV^-2.
        Used for scaling pseudo-data and estimating discovery sensitivity.
        """
        coeffs = self.eft_coefficients()
        return max(abs(v) for v in coeffs.values())

    def __repr__(self):
        return f"ZPrimeModel(gZH={self.gZH}, gZl={self.gZl}, mZp={self.mZp} TeV)"
