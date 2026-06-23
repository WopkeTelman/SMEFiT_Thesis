"""
models/wprime_constrained_v3.py

W' boson model — 14 truly constrained operators.

Derived from WPrimeConstrainedModel (16 ops) by removing the 2 operators
that remained prior-dominated in the free-EFT NS closure (50 L1 replicas,
gWH=0.50, mWp=5 TeV): Oll1133 and Oll1331 (std of pulls ≤ 0.3).

Removed vs WPrimeConstrainedModel:
    Oll1133, Oll1331    — 1st–3rd gen lepton mixing, prior-dominated in free-EFT fit
                          despite passing Fisher matrix test in UV-coupled context

Kept (14 constrained operators):
    O3pQ3, O3pl1, O3pl2, O3pl3
    OQl13, OQl1M
    Obp, Oll1111, Oll1122, Oll1221
    OpBox, OpQM, Otap, Otp
"""

from dataclasses import dataclass
from typing import Dict
from models.wprime import WPrimeModel


@dataclass
class WPrimeConstrainedV3Model(WPrimeModel):
    """
    W' model with 14 empirically constrained operators (free-EFT NS closure, v3).
    """

    OPERATORS = [
        "O3pQ3", "O3pl1", "O3pl2", "O3pl3",
        "OQl13", "OQl1M",
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
        return (f"WPrimeConstrainedV3Model(gWH={self.gWH}, gWLf={self.gWLf11:.4f}, "
                f"gWqf={self.gWqf33:.4f}, mWp={self.mWp} TeV) [14 ops]")
