from theory_vs_data import *

datasets = ['FCCee_Zdata']

BSM_path = f'/home/roan/smefit_bachelor_project/BSM_closure_tests/projections/wprime_constrained_gwh128_mwp093' # Path to the W' BSM projection
SM_path = '/home/roan/smefit_bachelor_project/smefit_database/commondata_projections_L0' # Path to the SM projection
theory_path = '/home/roan/smefit_bachelor_project/smefit_database/theory'

title = 'Zdata at FCCee'
# description: EWPOs at FCCee. The ordering is GammaZ (GeV), SigmaHad (pb), Re, Rmu, Rtau, Ae, Amu, Atau, Rb, Rc, Ab, Ac, alphaEW(mZ)
x_labels = [r'$\Gamma_Z$ (GeV)', r'$\sigma_{had}$ (pb)', r'$R_e$', r'$R_\mu$', r'$R_\tau$', r'$A_e$', r'$A_\mu$', r'$A_\tau$', r'$R_b$', r'$R_c$', r'$A_b$', r'$A_c$', r'$\alpha_{EW}(m_Z)$']

plot_filename = f'Zdata_comparison'
note = f'with W\' mass 9.3 TeV and gWH=1.28'

create_comparison_plot(datasets, SM_path, BSM_path, theory_path,
                        title=title,
                        x_labels=x_labels,
                        plot_filename=plot_filename,
                        modification_note=note, label_rotation=70, x_font_size=16, error_type='aggressive', log_scale=True)