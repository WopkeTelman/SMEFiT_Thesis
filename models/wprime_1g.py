"""
models/wprime_1g.py

W' fully-universal model: single free UV parameter g.

All couplings are equal:
    gWH = gWLf11 = gWLf22 = gWLf33 = gWqf33 = g

Every Wilson coefficient is then strictly proportional to g^2 / mWp^2,
so the model has only one free UV parameter (g) and mWp is fixed.

This is the maximally constrained single-coupling scenario.  Because there
are no mixed terms between independent couplings, impose_constrain in the
SMEFiT Fisher calculator is exact — the normalised Fisher heatmap from the
smefit report is correct without any truth-point correction.

Prefactors derived by substituting gWH = gWLf11 = gWLf22 = gWLf33 = gWqf33 = g
into WPrimeModel.eft_coefficients().
"""

from dataclasses import dataclass
from typing import Dict
from models.wprime import WPrimeModel
from models.wprime_constrained import WPrimeConstrainedModel


# Wilson_coeff = _1G_PREFACTOR[op] * g^2 / mWp^2
_1G_PREFACTOR = {
    # pure gWH^2
    "OpBox":   -0.375,
    "Obp":     -0.006002165432052166,
    "Otp":     -0.2480703588615627,
    "Otap":    -0.0025559460452279554,
    "Op":      -0.12938347743146458,
    # gWH * gWqf  →  g^2
    "OpQM":    +0.25,
    "O3pQ3":   -0.25,
    # gWH * gWLf  →  g^2
    "O3pl1":   -0.25,
    "O3pl2":   -0.25,
    "O3pl3":   -0.25,
    # gWLf * gWqf  →  g^2
    "OQl13":   -1.0,
    "OQl1M":   +1.0,
    "OQl23":   -1.0,
    "OQl2M":   +1.0,
    "OQl33":   -1.0,
    "OQl3M":   +1.0,
    # gWqf^2  →  g^2
    "OQQ1":    +0.08333333333333333,
    "OQQ8":    -1.0,
    # gWLf^2  →  g^2
    "Oll1111": -0.125,
    "Oll1122": +0.125,
    "Oll1133": +0.125,
    "Oll1221": -0.25,
    "Oll1331": -0.25,
    "Oll2222": -0.125,
    "Oll2233": +0.125,
    "Oll2332": -0.25,
    "Oll3333": -0.125,
}


@dataclass
class WPrime1gModel:
    """W' model with a single universal coupling g = gWH = gWLf = gWqf."""

    g:   float = 0.12
    mWp: float = 1.0

    # FCC-ee constrained operators only (same set as WPrimeConstrainedModel)
    OPERATORS = WPrimeConstrainedModel.OPERATORS
    ALL_OPS   = OPERATORS

    def eft_coefficients(self) -> Dict[str, float]:
        inner = WPrimeModel(
            gWH=self.g,
            gWLf11=self.g, gWLf22=self.g, gWLf33=self.g,
            gWqf33=self.g, mWp=self.mWp,
        )
        all_coeffs = inner.eft_coefficients()
        return {op: all_coeffs[op] for op in self.OPERATORS if op in all_coeffs}

    def uv_param_names(self):
        return ["g"]

    def uv_truth(self) -> Dict[str, float]:
        return {"g": self.g}

    def uv_coeff_block(self, prior_scale: float = 5.0) -> Dict:
        """Smefit coefficients block: every operator = prefactor * g^2 * mWp_TeV^{-2}."""
        prior_max = prior_scale * self.g
        block = {}
        for op in self.OPERATORS:
            c = _1G_PREFACTOR[op]
            block[op] = {"constrain": [{"g": [c, 2], "mWp_TeV": [1.0, -2]}]}
        block["g"]       = {"min": 0.0, "max": float(prior_max)}
        block["mWp_TeV"] = {"constrain": True, "value": float(self.mWp)}
        return block

    def __repr__(self):
        return (f"WPrime1gModel(g={self.g}, mWp={self.mWp} TeV) "
                f"[1 free UV param, {len(self.OPERATORS)} ops]")
