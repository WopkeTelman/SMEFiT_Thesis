"""
models/wprime_universal.py

W' fully-equal model: single free UV parameter gWH.

All fermion couplings are fixed to gWH (equal-coupling scenario):
    gWLf11 = gWLf22 = gWLf33 = gWqf33 = gWH

With this constraint every Wilson coefficient becomes proportional to
gWH^2 / mWp^2 with the same prefactor as a fully-universal W'.

Only gWH is sampled by the NS; mWp is fixed at the truth value.
ndof = n_data - 1 = 73 - 1 = 72.
"""

from dataclasses import dataclass
from typing import Dict
from models.wprime import WPrimeModel
from models.wprime_constrained import WPrimeConstrainedModel  # for uv_coeff_block operator list

# Prefactors c such that Wilson_coeff = c * gWH^2 / mWp^2
# Derived by substituting gWLf11=gWLf22=gWLf33=gWqf33 = gWH into
# the matching relations in WPrimeModel.eft_coefficients().
_UNIVERSAL_PREFACTOR = {
    # gWH-only operators (unchanged from WPrimeModel)
    "OpBox":   -0.375,
    "Obp":     -0.006002165432052166,
    "Otp":     -0.2480703588615627,
    "Otap":    -0.0025559460452279554,
    "Op":      -0.12938347743146458,
    # gWH * gWqf = gWH * gWH  →  gWH^2
    "OpQM":    +0.25,
    "O3pQ3":   -0.25,
    # gWH * gWLf = gWH * gWH  →  gWH^2
    "O3pl1":   -0.25,
    "O3pl2":   -0.25,
    "O3pl3":   -0.25,
    # gWLf * gWqf = gWH^2
    "OQl13":   -1.0,
    "OQl1M":   +1.0,
    "OQl23":   -1.0,
    "OQl2M":   +1.0,
    "OQl33":   -1.0,
    "OQl3M":   +1.0,
    # gWqf^2 = gWH^2
    "OQQ1":    +0.08333333333333333,
    "OQQ8":    -1.0,
    # gWLf^2 = gWH^2
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
class WPrimeUniversalModel:
    """W' model with only gWH as free UV parameter (gauge-universal scenario)."""

    gWH: float = 0.12
    mWp: float = 1.0

    # Full 27-operator set — matches the PCA K matrix built by the main pipeline.
    # The NS runcard constrains only the 15 FCC-ee-sensitive operators via
    # uv_coeff_block(); the remaining 12 are zero for any gWH value.
    OPERATORS = WPrimeModel.OPERATORS   # all 27
    ALL_OPS   = OPERATORS

    def _gf(self) -> float:
        return self.gWH

    def eft_coefficients(self) -> Dict[str, float]:
        inner = WPrimeModel(
            gWH=self.gWH,
            gWLf11=self._gf(), gWLf22=self._gf(), gWLf33=self._gf(),
            gWqf33=self._gf(), mWp=self.mWp,
        )
        return inner.eft_coefficients()   # all 27 operators

    def uv_param_names(self):
        return ["gWH"]

    def uv_truth(self) -> Dict[str, float]:
        return {"gWH": self.gWH}

    def uv_coeff_block(self, prior_scale: float = 5.0) -> Dict:
        """
        Smefit coefficients block for the universal NS runcard.
        Every operator is expressed as prefactor * gWH^2 * mWp_TeV^{-2}.
        """
        prior_max = prior_scale * self.gWH
        block = {}
        for op in self.OPERATORS:   # all 27
            c = _UNIVERSAL_PREFACTOR[op]
            block[op] = {"constrain": [{"gWH": [c, 2], "mWp_TeV": [1.0, -2]}]}
        block["gWH"]     = {"min": 0.0, "max": float(prior_max)}
        block["mWp_TeV"] = {"constrain": True, "value": float(self.mWp)}
        return block

    def __repr__(self):
        return (f"WPrimeUniversalModel(gWH={self.gWH}, "
                f"gWLf=gWqf=gWH={self._gf():.4f}, mWp={self.mWp} TeV) "
                f"[1 free UV param, equal couplings]")
