"""
models/wprime.py

W' boson model definition for the BSM discovery pipeline.

Matching relations: W' at mass mWp with couplings gWH (Higgs-W), gWLf (lepton), gWqf (quark)
at scale mWp, matched onto Warsaw basis SMEFT operators at 1/mWp^2.

Reference: hep-ph/XXXX (FCC-ee W' paper)

Usage:
    from models.wprime import WPrimeModel
    model = WPrimeModel(gWH=0.12, gWLf=0.04, gWqf=0.04, mWp=1.0)
    coeffs = model.eft_coefficients()   # dict: op -> value
    coeff_block = model.uv_coeff_block() # smefit runcard coefficients block
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class WPrimeModel:
    """
    W' boson UV model.

    Parameters
    ----------
    gWH    : W'-Higgs-W coupling
    gWLf11 : W'-lepton11 coupling (generation 1)
    gWLf22 : W'-lepton22 coupling (generation 2)
    gWLf33 : W'-lepton33 coupling (generation 3)
    gWqf33 : W'-quark33 coupling (generation 3)
    mWp    : W' mass in TeV
    """
    gWH:    float = 0.12
    gWLf11: float = 0.04
    gWLf22: float = 0.04
    gWLf33: float = 0.04
    gWqf33: float = 0.04
    mWp:    float = 1.0   # TeV

    # Operators generated at the matching scale (subset of Warsaw basis)
    OPERATORS = [
        "O3pQ3","O3pl1","O3pl2","O3pl3","OQQ1","OQQ8","OQl13","OQl1M",
        "OQl23","OQl2M","OQl33","OQl3M","Obp","Oll1111","Oll1122","Oll1133",
        "Oll1221","Oll1331","Oll2222","Oll2233","Oll2332","Oll3333","Op",
        "OpBox","OpQM","Otap","Otp"
    ]

    # Full list of 27D operators for smefit fits
    ALL_OPS = OPERATORS

    def eft_coefficients(self) -> Dict[str, float]:
        """
        Return EFT Wilson coefficients at the matching scale mWp.
        All coefficients in units of 1/mWp^2 [TeV^-2].

        Matching relations derived from tree-level W' exchange.
        """
        g    = self.gWH
        lf11 = self.gWLf11
        lf22 = self.gWLf22
        lf33 = self.gWLf33
        qf33 = self.gWqf33
        m2   = self.mWp**2

        return {
            "OpBox":   (-0.375 * g**2) / m2,
            "OpQM":    (0.25  * g * qf33) / m2,
            "O3pQ3":   (-0.25 * g * qf33) / m2,
            "O3pl1":   (-0.25 * g * lf11) / m2,
            "O3pl2":   (-0.25 * g * lf22) / m2,
            "O3pl3":   (-0.25 * g * lf33) / m2,
            "OQQ1":    (+0.08333333333333333 * qf33**2) / m2,
            "OQQ8":    (-1.0   * qf33**2) / m2,
            "Oll1111": (-0.125 * lf11**2) / m2,
            "Oll1122": (+0.125 * lf11 * lf22) / m2,
            "Oll1133": (+0.125 * lf11 * lf33) / m2,
            "Oll1221": (-0.25  * lf11 * lf22) / m2,
            "Oll1331": (-0.25  * lf11 * lf33) / m2,
            "Oll2222": (-0.125 * lf22**2) / m2,
            "Oll2233": (+0.125 * lf22 * lf33) / m2,
            "Oll2332": (-0.25  * lf22 * lf33) / m2,
            "Oll3333": (-0.125 * lf33**2) / m2,
            "OQl13":   (-1.0   * lf11 * qf33) / m2,
            "OQl1M":   (+1.0   * lf11 * qf33) / m2,
            "OQl23":   (-1.0   * lf22 * qf33) / m2,
            "OQl2M":   (+1.0   * lf22 * qf33) / m2,
            "OQl33":   (-1.0   * lf33 * qf33) / m2,
            "OQl3M":   (+1.0   * lf33 * qf33) / m2,
            "Obp":     (-0.006002165432052166  * g**2) / m2,
            "Otp":     (-0.2480703588615627    * g**2) / m2,
            "Otap":    (-0.0025559460452279554 * g**2) / m2,
            "Op":      (-0.12938347743146458   * g**2) / m2,
        }

    def uv_coeff_block(self, prior_scale: float = 5.0) -> Dict:
        """
        Return smefit UV coupling COEFF_BLOCK for NS runcard.

        Parameters
        ----------
        prior_scale : prior range = [0, prior_scale * gWH]
        """
        prior_max = prior_scale * self.gWH
        return {
            "OpBox":   {"constrain": [{"gWH": [-0.375, 2], "mWp_TeV": [1.0, -2]}]},
            "OpQM":    {"constrain": [{"gWH": [0.25, 1], "gWqf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "O3pQ3":   {"constrain": [{"gWH": [-0.25, 1], "gWqf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "O3pl1":   {"constrain": [{"gWH": [-0.25, 1], "gWLf11": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "O3pl2":   {"constrain": [{"gWH": [-0.25, 1], "gWLf22": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "O3pl3":   {"constrain": [{"gWH": [-0.25, 1], "gWLf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "OQQ1":    {"constrain": [{"gWH": [0.08333333333333333, 0], "gWqf33": [1, 2], "mWp_TeV": [1.0, -2]}]},
            "OQQ8":    {"constrain": [{"gWH": [-1.0, 0], "gWqf33": [1, 2], "mWp_TeV": [1.0, -2]}]},
            "Oll1221": {"constrain": [{"gWH": [-0.25, 0], "gWLf11": [1, 1], "gWLf22": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "Oll1111": {"constrain": [{"gWH": [-0.125, 0], "gWLf11": [1, 2], "mWp_TeV": [1.0, -2]}]},
            "Obp":     {"constrain": [{"gWH": [-0.006002165432052166, 2], "mWp_TeV": [1.0, -2]}]},
            "Otp":     {"constrain": [{"gWH": [-0.2480703588615627, 2], "mWp_TeV": [1.0, -2]}]},
            "Otap":    {"constrain": [{"gWH": [-0.0025559460452279554, 2], "mWp_TeV": [1.0, -2]}]},
            "Op":      {"constrain": [{"gWH": [-0.12938347743146458, 2], "mWp_TeV": [1.0, -2]}]},
            "OQl13":   {"constrain": [{"gWH": [-1.0, 0], "gWLf11": [1, 1], "gWqf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "OQl1M":   {"constrain": [{"gWH": [1.0, 0], "gWLf11": [1, 1], "gWqf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "OQl23":   {"constrain": [{"gWH": [-1.0, 0], "gWLf22": [1, 1], "gWqf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "OQl2M":   {"constrain": [{"gWH": [1.0, 0], "gWLf22": [1, 1], "gWqf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "Oll1122": {"constrain": [{"gWH": [0.125, 0], "gWLf11": [1, 1], "gWLf22": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "Oll1133": {"constrain": [{"gWH": [0.125, 0], "gWLf11": [1, 1], "gWLf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "Oll1331": {"constrain": [{"gWH": [-0.25, 0], "gWLf11": [1, 1], "gWLf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "Oll2222": {"constrain": [{"gWH": [-0.125, 0], "gWLf22": [1, 2], "mWp_TeV": [1.0, -2]}]},
            "Oll2332": {"constrain": [{"gWH": [-0.25, 0], "gWLf22": [1, 1], "gWLf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "Oll2233": {"constrain": [{"gWH": [0.125, 0], "gWLf22": [1, 1], "gWLf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "Oll3333": {"constrain": [{"gWH": [-0.125, 0], "gWLf33": [1, 2], "mWp_TeV": [1.0, -2]}]},
            "OQl3M":   {"constrain": [{"gWH": [1.0, 0], "gWLf33": [1, 1], "gWqf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "OQl33":   {"constrain": [{"gWH": [-1.0, 0], "gWLf33": [1, 1], "gWqf33": [1, 1], "mWp_TeV": [1.0, -2]}]},
            "gWH":     {"min": 0.0, "max": float(prior_max)},
            "gWLf11":  {"min": 0.0, "max": float(prior_max)},
            "gWLf22":  {"min": 0.0, "max": float(prior_max)},
            "gWLf33":  {"min": 0.0, "max": float(prior_max)},
            "gWqf33":  {"min": 0.0, "max": float(prior_max)},
            "mWp_TeV": {"constrain": True, "value": float(self.mWp)},
        }

    def uv_param_names(self):
        return ["gWH", "gWLf11", "gWLf22", "gWLf33", "gWqf33"]

    def uv_truth(self) -> Dict[str, float]:
        return {
            "gWH":    self.gWH,
            "gWLf11": self.gWLf11,
            "gWLf22": self.gWLf22,
            "gWLf33": self.gWLf33,
            "gWqf33": self.gWqf33,
        }

    def __repr__(self):
        return (f"WPrimeModel(gWH={self.gWH}, gWLf={self.gWLf11:.4f}, "
                f"gWqf={self.gWqf33:.4f}, mWp={self.mWp} TeV)")
