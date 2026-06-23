"""
models/zprime_constrained.py

Z' boson model — constrained operator set only.

Identical to ZPrimeModel but with the 2 flat directions (unconstrained by
FCC-ee datasets) removed from OPERATORS:

Removed (unconstrained, flat directions):
    Oll2222    — 4-muon (same-generation), no dedicated process at FCC-ee
    Oll3333    — 4-tau (same-generation), not independently constrained
    Oll2332    — mu-tau off-diagonal, no mu-tau mixed final state at FCC-ee

Kept (7 constrained operators):
    OpD                    — T-parameter (Higgs kinetic mixing)
    Opl1, Opl2, Opl3      — SU(2)-singlet Higgs-lepton coupling, all generations
    Oll1111                — 4-electron, constrained by e+e- → e+e-
    Oll1221                — mu-e mixed 4-lepton (off-diagonal)
    Oll1331                — tau-e mixed 4-lepton (off-diagonal)

Flat directions confirmed by Fisher matrix diagonal = 0 for Oll2222, Oll3333,
Oll2332 in the 50-dataset FCC-ee projection.
"""

from dataclasses import dataclass
from typing import Dict
from models.zprime import ZPrimeModel


@dataclass
class ZPrimeConstrainedModel(ZPrimeModel):
    """
    Z' model with only FCC-ee constrained operators.
    Flat directions (Oll2222, Oll3333) are excluded.
    """

    OPERATORS = [
        "OpD",
        "Opl1", "Opl2", "Opl3",
        "Oll1111", "Oll1221", "Oll1331",
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
                if k in self.OPERATORS or k in self.uv_param_names() + ["mZp_TeV"]}

    def __repr__(self):
        return (f"ZPrimeConstrainedModel(gZH={self.gZH}, gZl={self.gZl:.4f}, "
                f"mZp={self.mZp} TeV) [7 ops]")
