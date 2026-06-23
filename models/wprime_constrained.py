"""
models/wprime_constrained.py

W' boson model — constrained operator set only.

Identical to WPrimeModel but with the 11 flat directions (unconstrained by
FCC-ee datasets) removed from OPERATORS:

Removed (unconstrained):
    OQQ1, OQQ8          — 4-quark (3rd gen), no pure quark process at FCC-ee
    OQl23, OQl2M        — quark-muon lepton, not independently constrained
    OQl33, OQl3M        — quark-tau lepton, 3rd gen quarks barely produced
    Oll2222, Oll2233,
    Oll2332, Oll3333    — 4-muon/tau, not independently constrained
    Op                  — Higgs phi^4 kinetic, flat at FCC-ee energies

Kept (16 constrained operators):
    O3pQ3, O3pl1, O3pl2, O3pl3
    OQl13, OQl1M
    Obp, Oll1111, Oll1122, Oll1133, Oll1221, Oll1331
    OpBox, OpQM, Otap, Otp

Note: Oll1133 is confirmed constrained by Fisher matrix analysis (sig_c = 1.59e-4).
The earlier claim of Fierz degeneracy with Oll1331 was incorrect — both are
independently constrained by FCC-ee observables.
"""

from dataclasses import dataclass
from typing import Dict
from models.wprime import WPrimeModel


@dataclass
class WPrimeConstrainedModel(WPrimeModel):
    """
    W' model with only FCC-ee constrained operators.
    Flat directions (rank-deficient Fisher matrix entries) are excluded.
    """

    OPERATORS = [
        "O3pQ3", "O3pl1", "O3pl2", "O3pl3",
        "OQl13", "OQl1M",
        "Obp", "Oll1111", "Oll1122", "Oll1133", "Oll1221", "Oll1331",
        "OpBox", "OpQM", "Otap", "Otp",
    ]

    ALL_OPS = OPERATORS

    def eft_coefficients(self) -> Dict[str, float]:
        """Return only the constrained EFT coefficients."""
        all_coeffs = super().eft_coefficients()
        return {op: all_coeffs[op] for op in self.OPERATORS if op in all_coeffs}

    def uv_coeff_block(self, prior_scale: float = 5.0) -> Dict:
        """UV coupling block with only constrained operators."""
        full_block = super().uv_coeff_block(prior_scale=prior_scale)
        return {k: v for k, v in full_block.items()
                if k in self.OPERATORS or k in self.uv_param_names() + ["mWp_TeV"]}

    def __repr__(self):
        return (f"WPrimeConstrainedModel(gWH={self.gWH}, gWLf={self.gWLf11:.4f}, "
                f"gWqf={self.gWqf33:.4f}, mWp={self.mWp} TeV) [16 ops]")
