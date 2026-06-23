"""
models/wprime_constrained_v2.py

W' boson model — empirically constrained operator set.

Identical to WPrimeModel but with the 9 operators that are prior-dominated
in free-EFT NS fits (std of pulls ≤ 0.3 across 50 L1 replicas at
gWH=0.50, mWp=5 TeV) removed. Determined from ns_l1_closure.py --smeft run.

Removed (prior-dominated in free-EFT fit):
    OQQ1, OQQ8          — 4-quark (3rd gen), no pure quark process at FCC-ee
    Oll1133, Oll1331    — 1st–3rd gen lepton mixing, flat in free-EFT
    Oll2222, Oll2233,
    Oll2332, Oll3333    — 4-muon/tau, not independently constrained
    Op                  — Higgs phi^4 kinetic, flat at FCC-ee energies

Kept (18 constrained operators, empirically determined):
    O3pQ3, O3pl1, O3pl2, O3pl3
    OQl13, OQl1M, OQl23, OQl2M, OQl33, OQl3M
    Obp, Oll1111, Oll1122, Oll1221
    OpBox, OpQM, Otap, Otp
"""

from dataclasses import dataclass
from typing import Dict
from models.wprime import WPrimeModel


@dataclass
class WPrimeConstrainedV2Model(WPrimeModel):
    """
    W' model with 18 empirically constrained operators (free-EFT NS closure).
    """

    OPERATORS = [
        "O3pQ3", "O3pl1", "O3pl2", "O3pl3",
        "OQl13", "OQl1M", "OQl23", "OQl2M", "OQl33", "OQl3M",
        "Obp", "Oll1111", "Oll1122", "Oll1221",
        "OpBox", "OpQM", "Otap", "Otp",
    ]

    ALL_OPS = OPERATORS

    def eft_coefficients(self) -> Dict[str, float]:
        all_coeffs = super().eft_coefficients()
        return {op: all_coeffs[op] for op in self.OPERATORS if op in all_coeffs}

    def uv_coeff_block(self, prior_scale: float = 5.0) -> Dict:
        full_block = super().uv_coeff_block(prior_scale=prior_scale)
        return {k: v for k, v in full_block.items()
                if k in self.OPERATORS or k in self.uv_param_names() + ["mWp_TeV"]}

    def __repr__(self):
        return (f"WPrimeConstrainedV2Model(gWH={self.gWH}, gWLf={self.gWLf11:.4f}, "
                f"gWqf={self.gWqf33:.4f}, mWp={self.mWp} TeV) [18 ops]")
