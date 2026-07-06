# -*- coding: utf-8 -*-
import copy
import pathlib

import numpy as np
import pandas as pd
import yaml

from ..coefficients import CoefficientManager
from ..fit_manager import FitManager
from ..log import logging
from .chi2_utils import Chi2tableCalculator
from .coefficients_utils import CoefficientsPlotter, compute_confidence_level
from .correlations import plot_correlations
from .data_vs_theory import DataVsTheoryPlotter
from .discovery_utils import (
    plot_boundary_overlay,
    plot_discover_probability_curve,
    plot_fingerprint_classification,
    plot_representative_histograms,
)
from .bsm_sm_ratio import BsmSmRatioPlotter
from .fisher import FisherCalculator
from .html_utils import html_link, write_html_container
from .latex_tools import compile_tex
from .pca import PcaCalculator
from .summary import SummaryWriter

_logger = logging.getLogger(__name__)


class Report:
    r"""Report class manager.

    If :math:`\chi^2`, Fisher or Data vs Theory plots are produced it computes the
    best fit theory predictions.

    Attributes
    ----------
    report: str
        path to report folder
    fits: numpy.ndarray
        array with fits (instances of `smefit.fit_manager.FitManger`) included in the report
    data_info: pandas.DataFrame
        datasets information (references and data groups)
    coeff_info: pandas.DataFrame
        coefficients information (group and latex name)

    Parameters
    ----------
    report_path: pathlib.Path, str
        path to base folder, where the reports will be stored.
    result_path: pathlib.Path, str
        path to base folder, where the results are stored.
    report_config: dict
        dictionary with report configuration, see `/run_cards/analyze/report_runcard.yaml`
        for an example

    """

    def __init__(self, report_path, result_path, report_config):
        self.report = pathlib.Path(f"{report_path}/{report_config['name']}").absolute()
        self.fits = []
        # build the fits labels if needed
        result_ids = report_config.get("result_IDs", [])
        if "fit_labels" not in report_config:
            fit_labels = [
                f"${{\\rm {fit}}}$".replace("_", r"\ ")
                for fit in result_ids
            ]
        else:
            fit_labels = report_config["fit_labels"]
        # Loads fits
        for name, label in zip(result_ids, fit_labels):
            fit = FitManager(result_path, name, label)
            fit.load_results()

            if any(
                k in report_config
                for k in ["chi2_plots", "PCA", "fisher", "data_vs_theory"]
            ):
                fit.load_datasets()

            self.fits.append(fit)
        self.fits = np.array(self.fits)

        # Get names of datasets for each fit
        self.dataset_fits = []
        for fit in self.fits:
            self.dataset_fits.append([data["name"] for data in fit.config["datasets"]])

        # Loads useful information about data
        self.data_info = self._load_grouped_data_info(report_config.get("data_info", {}))
        # Loads coefficients grouped with latex name
        self.coeff_info = self._load_grouped_coeff_info(report_config.get("coeff_info", {}))
        self.html_index = ""
        self.html_content = ""

    def _load_grouped_data_info(self, raw_dict):
        """Load grouped info of datasets.

        Only elements appearing at least once in the fit configs are kept.

        Parameters
        ----------
        raw_dict: dict
            raw dictionary with relevant information

        Returns
        _______
        grouped_config: pandas.DataFrame
            table with information by group

        """
        out_dict = {}
        for group, entries in raw_dict.items():
            out_dict[group] = {}
            for val in entries:
                if np.any([val[0] in datasets for datasets in self.dataset_fits]):
                    out_dict[group][val[0]] = val[1]

            if len(out_dict[group]) == 0:
                out_dict.pop(group)
        if not out_dict:
            return pd.DataFrame()
        return pd.DataFrame(out_dict).stack().swaplevel()

    def _load_grouped_coeff_info(self, raw_dict):
        """Load grouped info of coefficients.

        Only elements appearing at least once in the fit configs are kept.

        Parameters
        ----------
        raw_dict: dict
            raw dictionary with relevant information

        Returns
        _______
        grouped_config: pandas.DataFrame
            table with information by group

        """
        out_dict = {}
        for group, entries in raw_dict.items():
            out_dict[group] = {}
            for val in entries:
                if np.any([val[0] in fit.config["coefficients"] for fit in self.fits]):
                    out_dict[group][val[0]] = val[1]

            if len(out_dict[group]) == 0:
                out_dict.pop(group)
        if not out_dict:
            return pd.DataFrame()
        return pd.DataFrame(out_dict).stack().swaplevel()

    def _append_section(self, title, links=None, figs=None, tables=None):
        self.html_index += html_link(f"#{title}", title, add_meta=False)
        self.html_content += write_html_container(
            title, links=links, figs=figs, dataFrame=tables
        )

    @staticmethod
    def _compute_closure_truth_from_coeff_config(coeff_cfg, coeff_names):
        """Compute closure-truth Wilson coefficients from a coefficient config."""
        fit_truth = {coeff_name: 0.0 for coeff_name in coeff_names}
        if not coeff_cfg:
            return fit_truth

        coefficients = CoefficientManager.from_dict(copy.deepcopy(coeff_cfg))
        if not coefficients.free_parameters.empty:
            # Free parameters have no closure-truth value by construction.
            coefficients.set_free_parameters(
                np.zeros(coefficients.free_parameters.shape[0], dtype=float)
            )
        coefficients.set_constraints()

        coeff_set = set(coefficients.name)
        for coeff_name in coeff_names:
            if coeff_name not in coeff_set:
                continue
            coeff_entry = coeff_cfg.get(coeff_name, {})
            if isinstance(coeff_entry, (int, float, np.floating)):
                fit_truth[coeff_name] = float(coeff_entry)
            elif isinstance(coeff_entry, dict) and (
                "value" in coeff_entry or "constrain" in coeff_entry
            ):
                fit_truth[coeff_name] = float(coefficients[coeff_name]["value"])
        return fit_truth

    def _load_closure_truth_points_from_runcards(
        self, closure_truth_runcards, coeff_names
    ):
        """Load closure-truth values from one or more projection/run runcards."""
        if isinstance(closure_truth_runcards, dict):
            runcard_paths = []
            for fit in self.fits:
                fit_runcard = closure_truth_runcards.get(fit.name, None)
                if fit_runcard is None:
                    raise ValueError(
                        "closure_truth_runcards is missing an entry for "
                        f"result_ID '{fit.name}'."
                    )
                runcard_paths.append(fit_runcard)
        elif isinstance(closure_truth_runcards, (str, pathlib.Path)):
            runcard_paths = [closure_truth_runcards for _ in self.fits]
        else:
            runcard_paths = list(closure_truth_runcards)
            if len(runcard_paths) == 1 and len(self.fits) > 1:
                runcard_paths = [runcard_paths[0] for _ in self.fits]
            elif len(runcard_paths) != len(self.fits):
                raise ValueError(
                    "closure_truth_runcards must have either 1 entry or one entry per "
                    f"fit ({len(self.fits)}). Got {len(runcard_paths)} entries."
                )

        closure_truth_points = []
        for runcard in runcard_paths:
            runcard_path = pathlib.Path(runcard).expanduser()
            with open(runcard_path, encoding="utf-8") as f:
                runcard_config = yaml.safe_load(f)
            coeff_cfg = runcard_config.get("coefficients", {})
            closure_truth_points.append(
                self._compute_closure_truth_from_coeff_config(coeff_cfg, coeff_names)
            )
        return closure_truth_points

    def _load_closure_truth_points_from_fit_configs(self, coeff_names):
        closure_truth_points = []
        for fit in self.fits:
            coeff_cfg = fit.config.get("coefficients", {})
            closure_truth_points.append(
                self._compute_closure_truth_from_coeff_config(coeff_cfg, coeff_names)
            )
        return closure_truth_points

    def summary(self):
        """Summary Table runner."""
        summary = SummaryWriter(self.fits, self.data_info, self.coeff_info)
        section_title = "Summary"
        coeff_tab = "coefficient_summary"
        data_tab = "dataset_summary"

        # write summary tables
        compile_tex(self.report, summary.write_coefficients_table(), coeff_tab)
        compile_tex(self.report, summary.write_dataset_table(), data_tab)

        self._append_section(
            section_title,
            links=[(data_tab, "Dataset summary"), (coeff_tab, "Coefficient summary")],
            tables=summary.fit_settings(),
        )

    def chi2(self, table=True, plot_experiment=None, plot_distribution=None):
        r""":math:`\chi^2` table and plots runner.

        Parameters
        ----------
        table: bool, optional
            write the latex :math:`\chi^2` table per dataset
        plot_experiment: bool, optional
            plot the :math:`\chi^2` per dataset
        plot_distribution: bool, optional
            plot the :math:`\chi^2` distribution per each replica

        """
        links_list = None
        figs_list = []
        chi2_cal = Chi2tableCalculator(self.data_info)

        # here we store the info for each fit
        chi2_dict = {}
        chi2_dict_group = {}
        chi2_replica = {}
        for fit in self.fits:
            # This computes the chi2 by taking the mean of the replicas
            _, chi2_total_rep = chi2_cal.compute(
                fit.datasets,
                fit.smeft_predictions,
            )

            chi2_df_best, _ = chi2_cal.compute(
                fit.datasets, fit.smeft_predictions_best_fit
            )

            chi2_replica[fit.label] = chi2_total_rep
            chi2_dict[fit.label] = chi2_cal.add_normalized_chi2(chi2_df_best)
            chi2_dict_group[fit.label] = chi2_cal.group_chi2_df(chi2_df_best)

        if table:
            lines = chi2_cal.write(chi2_dict, chi2_dict_group)
            compile_tex(self.report, lines, "chi2_tables")
            links_list = [("chi2_tables", "Tables")]

        if plot_experiment is not None:
            _logger.info("Plotting : chi^2 for each dataset")
            chi2_cal.plot_exp(chi2_dict, f"{self.report}/chi2_bar", **plot_experiment)
            figs_list.append("chi2_bar")

        if plot_distribution is not None:
            _logger.info("Plotting : chi^2 distribution for each replica")
            chi2_cal.plot_dist(
                chi2_replica, f"{self.report}/chi2_histo", **plot_distribution
            )
            figs_list.append("chi2_histo")

        self._append_section("Chi2", links=links_list, figs=figs_list)

    def data_vs_theory(
        self,
        fit_list=None,
        datasets=None,
        panel="pull",
        include_sm=True,
        include_best_fit=True,
        show_legend=True,
        title=True,
        figsize=(8, 5),
        per_dataset=True,
        show_dataset_boundaries=True,
    ):
        """Plot dataset-level comparisons of data against SM and best-fit theory."""
        figs_list = []

        if fit_list is not None:
            fit_list = [fit for fit in self.fits if fit in fit_list]
        else:
            fit_list = self.fits

        dataset_labels = {}
        for (_, dataset), label in self.data_info.items():
            if isinstance(label, str) and (
                label.startswith("http") or label.startswith("https")
            ):
                continue
            dataset_labels[dataset] = label

        dvst_plotter = DataVsTheoryPlotter(self.report, dataset_labels=dataset_labels)
        for fit in fit_list:
            _logger.info(f"Plotting data-vs-theory for fit: {fit.name}")
            figs_list.extend(
                dvst_plotter.plot_fit(
                    fit,
                    datasets_to_plot=datasets,
                    panel=panel,
                    include_sm=include_sm,
                    include_best_fit=include_best_fit,
                    show_legend=show_legend,
                    title=title,
                    figsize=figsize,
                    per_dataset=per_dataset,
                    show_dataset_boundaries=show_dataset_boundaries,
                )
            )

        self._append_section("Data vs Theory", figs=figs_list)

    def discovery(
        self,
        boundary_summary_csv,
        posterior_map_csv,
        representative_points_csv=None,
        posterior_samples_root=None,
        representative_point_tags=None,
        fingerprint_grid_csv=None,
        mass=1.0,
        pvalue_threshold=2.866515718791933e-07,
        boundary_order=None,
        make_fingerprint_plot=False,
    ):
        """Discovery-focused plots from precomputed robust-discovery CSV outputs."""
        figs_list = []

        boundary_path = pathlib.Path(boundary_summary_csv)
        post_map_path = pathlib.Path(posterior_map_csv)
        if not boundary_path.is_file():
            raise FileNotFoundError(f"Missing boundary summary file: {boundary_path}")
        if not post_map_path.is_file():
            raise FileNotFoundError(f"Missing posterior map file: {post_map_path}")

        boundary_df = pd.read_csv(boundary_path)
        post_map_df = pd.read_csv(post_map_path)

        overlay_name = "discovery_boundary_overlay"
        plot_boundary_overlay(
            boundary_df, f"{self.report}/{overlay_name}", boundary_order=boundary_order
        )
        figs_list.append(overlay_name)

        discover_prob_name = "discovery_prob_vs_kappa"
        plot_discover_probability_curve(
            post_map_df, f"{self.report}/{discover_prob_name}", mass=mass
        )
        figs_list.append(discover_prob_name)

        if representative_points_csv is not None:
            rep_path = pathlib.Path(representative_points_csv)
            if rep_path.is_file() and posterior_samples_root is not None:
                rep_df = pd.read_csv(rep_path)
                rep_figs = plot_representative_histograms(
                    rep_df,
                    posterior_samples_root=posterior_samples_root,
                    out_prefix=f"{self.report}/discovery_representative",
                    point_tag_map=representative_point_tags,
                    pvalue_threshold=pvalue_threshold,
                )
                figs_list.extend(rep_figs)
            else:
                _logger.warning(
                    "Skipping representative histograms: missing representative_points_csv "
                    "or posterior_samples_root."
                )

        if make_fingerprint_plot and fingerprint_grid_csv is not None:
            fp_path = pathlib.Path(fingerprint_grid_csv)
            if fp_path.is_file():
                fp_df = pd.read_csv(fp_path)
                fp_name = "discovery_fingerprint_classification"
                plot_fingerprint_classification(fp_df, f"{self.report}/{fp_name}")
                figs_list.append(fp_name)
            else:
                _logger.warning("Skipping fingerprint plot, missing file: %s", fp_path)

        self._append_section("Discovery", figs=figs_list, tables=boundary_df)

    def coefficients(
        self,
        scatter_plot=None,
        confidence_level_bar=None,
        pull_bar=None,
        spider_plot=None,
        posterior_histograms=True,
        contours_2d=None,
        hide_dofs=None,
        show_only=None,
        logo=True,
        table=None,
        double_solution=None,
    ):
        """Coefficients plots and table runner.

        Parameters
        ----------
        hide_dofs: list
            list of operator not to display
        show_only: list
            list of all the operator to display, if None all the free dof are presented
        logo: bool
            if True add logo to the plots
        scatter_plot: None, dict
            kwarg confidence level bar plot or None
        confidence_level_bar: None, dict
            kwarg scatter plot or None
        posterior_histograms: bool
            if True plot the posterior distribution for each coefficient
            Additional supported keys in the `posterior_histograms` block:
            - closure_truth_points: explicit list of dict values per fit.
            - closure_truth_runcards: path/list/dict to projection (or run) runcard(s)
              used to generate closure pseudo-data. Values are evaluated from the
              `coefficients` block, including constrained relations.
        table: None, dict
            kwarg the latex confidence level table per coefficient or None
        double_solution: dict
            operator with double solution per fit

        """
        links_list = None
        figs_list = []
        coeff_config = self.coeff_info
        if show_only is not None:
            coeff_config = coeff_config.loc[:, show_only]
        if hide_dofs is not None:
            coeff_config = coeff_config.drop(hide_dofs, level=1)

        coeff_plt = CoefficientsPlotter(
            self.report,
            coeff_config,
            logo=logo,
        )

        # compute confidence level bounds
        bounds_dict = {}
        for fit in self.fits:
            bounds_dict[fit.label] = compute_confidence_level(
                fit.results["samples"],
                coeff_plt.coeff_info,
                fit.has_posterior,
                (
                    double_solution.get(fit.name, None)
                    if double_solution is not None
                    else None
                ),
            )

        if scatter_plot is not None:
            _logger.info("Plotting : Central values and Confidence Level bounds")
            coeff_plt.plot_coeffs(bounds_dict, **scatter_plot)
            figs_list.append("coefficient_central")

        # when we plot the 95% CL we show the 95% CL for null solutions.
        # the error coming from a degenerate solution is not taken into account.
        if confidence_level_bar is not None:
            _logger.info("Plotting : Confidence Level error bars")
            bar_cl = confidence_level_bar["confidence_level"]
            confidence_level_bar.pop("confidence_level")
            zero_sol = 0
            coeff_plt.plot_coeffs_bar(
                {
                    name: -bound_df.loc[zero_sol, f"low{bar_cl}"]
                    + bound_df.loc[zero_sol, f"high{bar_cl}"]
                    for name, bound_df in bounds_dict.items()
                },
                **confidence_level_bar,
            )
            figs_list.append("coefficient_bar")

        # when we plot the 95% CL we show the 95% CL for null solutions.
        # the error coming from a degenerate solution is not taken into account.
        if pull_bar is not None:
            _logger.info("Plotting : Pull ")
            zero_sol = 0
            coeff_plt.plot_pull(
                {
                    name: bound_df.loc[zero_sol, "pull"]
                    for name, bound_df in bounds_dict.items()
                },
                **pull_bar,
            )
            figs_list.append("pull_bar")

        if spider_plot is not None:
            _logger.info("Plotting : spider plot")

            spider_cl = spider_plot["confidence_level"]
            spider_plot.pop("confidence_level")

            spider_bounds = {}
            for name, bound_df in bounds_dict.items():
                dbl_solution = bound_df.index.get_level_values(0)
                # if dbl solution requested, add the confidence intervals, otherwise just
                # use the sum of the hdi intervals
                if 1 in dbl_solution:
                    dbl_op = double_solution.get(fit.name, None)
                    idx = [
                        np.argwhere(
                            self.coeff_info.index.get_level_values(1) == op
                        ).flatten()[0]
                        for op in dbl_op
                    ]
                    bound_df_dbl = bound_df.iloc[:, idx]

                    width_0 = bound_df_dbl.loc[0, f"hdi_{spider_cl}"]
                    width_1 = bound_df_dbl.loc[1, f"hdi_{spider_cl}"]
                    width_tot = width_0 + width_1

                    # update bound df
                    bound_df.loc[0, f"hdi_{spider_cl}"].iloc[idx] = width_tot

                    spider_bounds[name] = bound_df.loc[0, f"hdi_{spider_cl}"]

                else:
                    spider_bounds[name] = bound_df.loc[0, f"hdi_{spider_cl}"]

            coeff_plt.plot_spider(
                spider_bounds,
                labels=[fit.label for fit in self.fits],
                **spider_plot,
            )
            figs_list.append("spider_plot")

        if posterior_histograms:
            _logger.info("Plotting : Posterior histograms")
            disjointed_lists = [
                (
                    double_solution.get(fit.name, None)
                    if double_solution is not None
                    else None
                )
                for fit in self.fits
            ]
            posterior_histograms["disjointed_lists"] = disjointed_lists
            closure_truth_points = posterior_histograms.pop(
                "closure_truth_points", None
            )
            closure_truth_runcards = posterior_histograms.pop(
                "closure_truth_runcards", None
            )
            if closure_truth_points is None:
                coeff_names = coeff_plt.coeff_info.index.get_level_values(1)
                if closure_truth_runcards is not None:
                    closure_truth_points = (
                        self._load_closure_truth_points_from_runcards(
                            closure_truth_runcards, coeff_names
                        )
                    )
                else:
                    closure_truth_points = (
                        self._load_closure_truth_points_from_fit_configs(coeff_names)
                    )

            coeff_plt.plot_posteriors(
                [fit.results["samples"] for fit in self.fits],
                labels=[fit.label for fit in self.fits],
                closure_truth_points=closure_truth_points,
                **posterior_histograms,
            )
            figs_list.append("coefficient_histo")

        if table is not None:
            _logger.info("Writing : Confidence level table")
            lines = coeff_plt.write_cl_table(bounds_dict, **table)
            compile_tex(self.report, lines, "coefficients_table")
            links_list = [("coefficients_table", "CL table")]

        if contours_2d:
            _logger.info("Plotting : 2D confidence level projections")
            coeff_plt.plot_contours_2d(
                [
                    (
                        fit.results["samples"][fit.coefficients.free_parameters.index],
                        fit.config["use_quad"],
                    )
                    for fit in self.fits
                ],
                labels=[fit.label for fit in self.fits],
                confidence_level=contours_2d["confidence_level"],
                dofs_show=contours_2d["dofs_show"],
                double_solution=double_solution,
            )
            figs_list.append("contours_2d")

        self._append_section("Coefficients", links=links_list, figs=figs_list)

    def bsm_sm_ratio(
        self,
        data_path,
        theory_path,
        datasets,
        coefficients,
        use_quad=False,
        use_theory_covmat=True,
        figsize=(12, 7),
    ):
        """Plot the cumulative SMEFT K-factor applied to each dataset.

        The K-factor is computed from the contamination coefficients specified
        directly in the runcard, without requiring any fit result.

        Parameters
        ----------
        data_path : str
            Path to the commondata folder.
        theory_path : str
            Path to the theory folder.
        datasets : list of dict
            List of dataset specs, e.g. ``[{name: FCCee_ww_161GeV, order: LO}]``.
        coefficients : dict
            Wilson coefficient values, e.g. ``{OpBox: -0.034, OpQM: 0.0075}``.
            Operators absent from this dict are set to zero.
        use_quad : bool
            Whether to include quadratic EFT corrections.
        use_theory_covmat : bool
            Whether to include the theory covariance matrix.
        figsize : list or tuple
            Figure size ``[width, height]`` in inches.
        """
        _logger.info("Plotting BSM/SM ratio from contamination parameters")
        plotter = BsmSmRatioPlotter(self.report)
        figs_list = plotter.plot(
            data_path=data_path,
            theory_path=theory_path,
            datasets=datasets,
            coefficients=coefficients,
            use_quad=use_quad,
            use_theory_covmat=use_theory_covmat,
            figsize=tuple(figsize),
        )
        self._append_section("BSM/SM Ratio", figs=figs_list)

    def correlations(
        self, hide_dofs=None, thr_show=0.1, title=True, fit_list=None, figsize=(10, 10)
    ):
        """Plot coefficients correlation matrix.

        Parameters
        ----------
        hide_dofs: list
            list of operator not to display.
        thr_show: float, None
            minimum threshold value to show.
            If None the full correlation matrix is displayed.
        title: bool
            if True display fit label name as title
        fit_list: list, optional
            list of fit names for which the correlation is computed.
            By default all the fits included in the report
        """
        figs_list = []

        if fit_list is not None:
            fit_list = [fit for fit in self.fits if fit in fit_list]
        else:
            fit_list = self.fits

        for fit in fit_list:
            _logger.info(f"Plotting correlations for: {fit.name}")
            coeff_to_keep = fit.coefficients.free_parameters.index
            plot_correlations(
                fit.results["samples"][coeff_to_keep],
                latex_names=self.coeff_info.droplevel(0),
                fig_name=f"{self.report}/correlations_{fit.name}",
                title=fit.label if title else None,
                hide_dofs=hide_dofs,
                thr_show=thr_show,
                figsize=figsize,
            )
            figs_list.append(f"correlations_{fit.name}")

        self._append_section("Correlations", figs=figs_list)

    def pca(
        self,
        table=True,
        plot=None,
        thr_show=1e-2,
        fit_list=None,
    ):
        """Principal Components Analysis runner.

        Parameters
        ----------
        table: bool, optional
            if True writes the PC directions in a latex list
        plot: bool, optional
            if True produces a PC heatmap
        thr_show: float
            minimum threshold value to show
        fit_list: list, optional
            list of fit names for which the PCA is computed.
            By default all the fits included in the report
        """
        figs_list, links_list = [], []
        if fit_list is not None:
            fit_list = [fit for fit in self.fits if fit in fit_list]
        else:
            fit_list = self.fits
        for fit in fit_list:
            _logger.info(f"Computing PCA for fit {fit.name}")
            pca_cal = PcaCalculator(
                fit.datasets,
                fit.coefficients,
                self.coeff_info.droplevel(0),
            )
            pca_cal.compute()

            if table:
                compile_tex(
                    self.report,
                    pca_cal.write(fit.label, thr_show),
                    f"pca_table_{fit.name}",
                )
                links_list.append((f"pca_table_{fit.name}", f"Table {fit.label}"))
            if plot is not None:
                title = fit.name

                # TODO: check why **fit_plot got removed (see PR)
                pca_cal.plot_heatmap(
                    f"{self.report}/pca_heatmap_{fit.name}",
                    title=title,
                    figsize=plot["figsize"],
                )
                figs_list.append(f"pca_heatmap_{fit.name}")
        self._append_section("PCA", figs=figs_list, links=links_list)

    def fisher(
        self, norm="coeff", summary_only=True, plot=None, fit_list=None, log=False
    ):
        """Fisher information table and plots runner.

        Summary table and plots are the default

        Parameters
        ----------
        norm: "coeff", "dataset"
            fisher information normalization: per coefficient, or per dataset
        summary_only: bool, optional
            if False writes the fine grained fisher tables per dataset and group
            if True only the summary table with grouped a datsets is written
        plot: None, dict
            plot options
        fit_list: list, optional
            list of fit names for which the fisher information is computed.
            By default all the fits included in the report
        log: bool, optional
            if True shows the log of the Fisher informaltion

        """
        figs_list, links_list = [], []
        if fit_list is not None:
            fit_list = [fit for fit in self.fits if fit in fit_list]
        else:
            fit_list = self.fits

        fishers = {}
        for fit in fit_list:
            compute_quad = fit.config["use_quad"]
            fisher_cal = FisherCalculator(fit.coefficients, fit.datasets, compute_quad)
            fisher_cal.compute_linear()
            fisher_cal.lin_fisher = fisher_cal.normalize(
                fisher_cal.lin_fisher, norm=norm, log=log
            )
            fisher_cal.summary_table = fisher_cal.groupby_data(
                fisher_cal.lin_fisher, self.data_info, norm, log
            )
            fishers[fit.name] = fisher_cal

            # if necessary compute the quadratic Fisher
            if compute_quad:
                fisher_cal.compute_quadratic(
                    fit.results["samples"], fit.smeft_predictions
                )
                fisher_cal.quad_fisher = fisher_cal.normalize(
                    fisher_cal.quad_fisher, norm=norm, log=log
                )
                fisher_cal.summary_HOtable = fisher_cal.groupby_data(
                    fisher_cal.quad_fisher, self.data_info, norm, log
                )

            compile_tex(
                self.report,
                fisher_cal.write_grouped(self.coeff_info, self.data_info, summary_only),
                f"fisher_{fit.name}",
            )
            links_list.append((f"fisher_{fit.name}", f"Table {fit.label}"))

            if plot is not None:
                fit_plot = copy.deepcopy(plot)
                fit_plot.pop("together", None)
                title = fit.label if fit_plot.pop("title") else None
                fisher_cal.plot_heatmap(
                    self.coeff_info,
                    f"{self.report}/fisher_heatmap_{fit.name}",
                    title=title,
                    **fit_plot,
                )
                figs_list.append(f"fisher_heatmap_{fit.name}")

        # plot both fishers
        if plot is not None and plot.get("together", False):
            fisher_1 = fishers[plot["together"][0]]
            fisher_2 = fishers[plot["together"][1]]
            fit_plot = copy.deepcopy(plot)
            fit_plot.pop("together")

            # show title of last fit
            title = fit.label if fit_plot.pop("title") else None

            # make heatmap of fisher_1 and fisher_2
            fisher_2.plot_heatmap(
                self.coeff_info,
                f"{self.report}/fisher_heatmap_both",
                title=title,
                other=fisher_1,
                labels=[fit.label for fit in self.fits],
                **fit_plot,
            )
            figs_list.append(f"fisher_heatmap_both")

        self._append_section("Fisher", figs=figs_list, links=links_list)
