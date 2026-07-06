"""
models/zprime.py

Z' boson model definition for the BSM discovery pipeline.

Model: B'-like (hypercharge-singlet) Z' with universal lepton coupling.

Lagrangian:
    L ⊃ +½ mZp² Z'μ Z'^μ
        + gZH  Z'μ (H† i D^μ H) + h.c.    [Higgs current]
        + gZl  Z'μ Σ_gen (l̄_L,gen γ^μ l_L,gen)  [universal left-handed lepton]

Key: This is NOT a sequential Z' (not same quantum numbers as SM Z).
     It is a SU(2)_L×U(1)_Y singlet, coupling only to the H current
     and left-handed lepton current → generates O_{Hl}^(1) not O_{Hl}^(3).

Tree-level matching (Warsaw basis, integrating out Z' at scale mZp):

  L_eff = -1/(2 mZp²) (gZH J^H + gZl J^l)²,   J^H_μ = H† i(D_μ - D←_μ) H

  Wilson coefficient formula: C_i = prefactor × g_a^p × g_b^q × mZp^(-2)

  1. Higgs-current square. Writing a_μ = H†D_μH, the square of the
     Hermitian Higgs current reduces ALGEBRAICALLY (no EOM needed):
        J^H · J^H = 4 a*·a - (a + a*)² = 4 O_HD + O_HBox   (after IBP)
     since a_μ + a*_μ = ∂_μ(H†H) and ∂(H†H)·∂(H†H) = -O_HBox up to a
     total derivative. Hence TWO bosonic operators:
        C_HD   = -2   × gZH² / mZp²      (T-parameter shift)
        C_HBox = -1/2 × gZH² / mZp²      (universal Higgs-coupling rescale)
     Cross-check: de Blas et al. dictionary, singlet vector B row:
        c_φD = -2|g^φ|²,  c_φ□ = -½|g^φ|².
     No O_H (H†H)³ and no Yukawa-like operators are generated: those
     arise only when the EOM must be used, as for the SU(2) triplet W'.

  2. Opl1,2,3  (O_{Hl}^(1) = (H† i↔D_μ H)(l̄_gen γ^μ l_gen)):
        C_{Hl,gen}^(1) = - gZH * gZl / mZp²
        Physical origin: cross term 2 gZH gZl J^H·J^l with the overall
        -1/(2m²) prefactor.

  3. Oll_iiii  (O_{ll}^{iiii}, i=1,2,3):
        C_{ll}^{iiii} = - gZl² / (2 mZp²)
        Same-generation diagonal term in (Σ_i l̄_i γ^μ l_i)²;
        factor 1/2 from the overall -1/(2m²) in L_eff.

  4. Oll1122, Oll1133, Oll2233  (DIRECT off-diagonal structures,
     O_{ll}^{iijj} = (l̄_i γ^μ l_i)(l̄_j γ_μ l_j), i<j):
        C_{ll}^{iijj} = - gZl² / (2 mZp²)
        Physical origin: cross terms 2 J^l_i · J^l_j in (Σ_i J^l_i)² with
        the overall -1/(2m²) prefactor give a physical term -gZl²/m²; the
        Warsaw full-sum convention (both (iijj) and (jjii) assignments carry
        the named coefficient) halves the quoted coefficient. Matches the
        de Blas dictionary (c_ll)_prst = -1/2 g^l_pr g^l_st, and the same
        convention reproduces the verified W' values A.8-A.10.

  IMPORTANT — no Fierz trade. For SU(2)_L DOUBLETS there is no ±1
  relation between O^{iijj} and O^{ijji}: the chiral Fierz identity
  (valid with a PLUS sign for anticommuting fields) crosses the SU(2)
  indices, giving
        O^{ijji} = 2 (l̄_i γ_μ T^a l_i)(l̄_j γ^μ T^a l_j) + ½ O^{iijj},
  i.e. the crossed structure is an independent combination involving
  the triplet contraction. The singlet Z' therefore populates ONLY the
  direct structures O^{iijj}; the crossed operators Oll1221/1331/2332
  are NOT generated (they belong to the triplet W' matching).

Distinct from W':
  W' generates: O3pl (SU(2) triplet), OQl (charged current), Yukawa-like
  Z' generates: OpD (T-parameter), Opl (SU(2) singlet), direct Oll only
  Shared:       OpBox (different prefactor: -½ g² vs -⅜ g²), diagonal Oll
  → different Warsaw-basis footprint

All operators confirmed available in smefit FCC-ee theory database.
See: smefit_database/theory/FCCee_*.json, key 'LO'.
(OpBox and Oll1122/1133/2233 are used by the W' model already.)

Reference: de Blas et al., arXiv:1711.10391 (SMEFT tree-level dictionary)
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

    Generates operators: OpD, OpBox, Opl1, Opl2, Opl3,
                         Oll1111, Oll2222, Oll3333,
                         Oll1122, Oll1133, Oll2233
    All confirmed available in smefit FCC-ee theory database.
    """
    gZH: float = 0.12
    gZl: float = 0.04
    mZp: float = 1.0   # TeV

    # Warsaw basis operators generated at tree-level by this Z' model (11)
    # All confirmed in smefit FCC-ee theory database (LO key)
    OPERATORS = [
        "OpD",      # O_{HD}: T-parameter shift, from (J^H)² via Z' exchange
        "OpBox",    # O_{H□}: from (J^H)², algebraic companion of OpD
        "Opl1",     # O_{Hl,1}^(1): Z'-Higgs-lepton_gen1, singlet H current
        "Opl2",     # O_{Hl,2}^(1): Z'-Higgs-lepton_gen2
        "Opl3",     # O_{Hl,3}^(1): Z'-Higgs-lepton_gen3
        "Oll1111",  # O_{ll}^{1111}: diagonal same-gen term gen1
        "Oll2222",  # O_{ll}^{2222}: diagonal same-gen term gen2
        "Oll3333",  # O_{ll}^{3333}: diagonal same-gen term gen3
        "Oll1122",  # O_{ll}^{1122}: direct cross term gen1×gen2
        "Oll1133",  # O_{ll}^{1133}: direct cross term gen1×gen3
        "Oll2233",  # O_{ll}^{2233}: direct cross term gen2×gen3
    ]

    # smefit operator names (same list — all confirmed in FCC-ee DB)
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
            # Bosonic sector: J^H·J^H = 4 O_HD + O_HBox  (algebraic, + IBP)
            # with overall -1/(2m²) prefactor from integrating out the Z'
            "OpD":     -2.0 * g**2 / m2,
            "OpBox":   -0.5 * g**2 / m2,

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

            # Four-lepton off-diagonal: DIRECT structures only.
            # Cross terms 2 J_i·J_j give a physical term -gZl²/m² O^{iijj};
            # in the Warsaw full-sum convention (C_prst summed over ALL index
            # assignments, so (iijj) and (jjii) each carry the named coeff)
            # the SMEFiT coefficient is HALF of that: -gZl²/(2 m²).
            # Anchor: the W' relations A.8-A.10 show the same pattern
            # (diagonal -1/8, off-diagonal +1/8 / -1/4 = naive/2), and the
            # de Blas dictionary gives (c_ll)_prst = -1/2 g^l_pr g^l_st.
            # No Fierz trade to ijji: invalid for SU(2) doublets (see header).
            "Oll1122": -0.5 * gl**2 / m2,
            "Oll1133": -0.5 * gl**2 / m2,
            "Oll2233": -0.5 * gl**2 / m2,
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

            # OpD = -2 × gZH² / mZp²
            "OpD":     {"constrain": [{"gZH": [-2.0, 2], "mZp_TeV": [1.0, -2]}]},

            # OpBox = -0.5 × gZH² / mZp²
            "OpBox":   {"constrain": [{"gZH": [-0.5, 2], "mZp_TeV": [1.0, -2]}]},

            # Opl_gen = -gZH × gZl × mZp^{-2}
            "Opl1":    {"constrain": [{"gZH": [-1.0, 1], "gZl": [1.0, 1], "mZp_TeV": [1.0, -2]}]},
            "Opl2":    {"constrain": [{"gZH": [-1.0, 1], "gZl": [1.0, 1], "mZp_TeV": [1.0, -2]}]},
            "Opl3":    {"constrain": [{"gZH": [-1.0, 1], "gZl": [1.0, 1], "mZp_TeV": [1.0, -2]}]},

            # Oll (diagonal) = -0.5 × gZl² × mZp^{-2}
            "Oll1111": {"constrain": [{"gZl": [-0.5, 2], "mZp_TeV": [1.0, -2]}]},
            "Oll2222": {"constrain": [{"gZl": [-0.5, 2], "mZp_TeV": [1.0, -2]}]},
            "Oll3333": {"constrain": [{"gZl": [-0.5, 2], "mZp_TeV": [1.0, -2]}]},

            # Oll (off-diagonal, direct iijj form) = -0.5 × gZl² × mZp^{-2}
            # (Warsaw full-sum convention; see eft_coefficients comment)
            "Oll1122": {"constrain": [{"gZl": [-0.5, 2], "mZp_TeV": [1.0, -2]}]},
            "Oll1133": {"constrain": [{"gZl": [-0.5, 2], "mZp_TeV": [1.0, -2]}]},
            "Oll2233": {"constrain": [{"gZl": [-0.5, 2], "mZp_TeV": [1.0, -2]}]},

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
