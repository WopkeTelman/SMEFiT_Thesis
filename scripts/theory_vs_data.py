import numpy as np
import matplotlib.pyplot as plt
import yaml
import json
import os

#filename = 'FCCee_Zdata.yaml'

def extract_data(path, dataset):
    with open(os.path.join(path, f'{dataset}.yaml'), 'r') as f:
        data_set = yaml.safe_load(f)

    # if data is not a list, make it a list
    data = np.array(data_set['data_central'])
    stat_error = np.array(data_set['statistical_error']) # Only possible when the errors are uncorrelated
    sys_error = np.diag(data_set['systematics'])

    error = np.sqrt(stat_error**2 + sys_error**2)

    return data, error

error_types = {
    'current': 'theory_cov_current',
    'conservative': 'theory_cov_conservative',
    'aggressive': 'theory_cov_aggressive'
}

def extract_theory(path, dataset, error_type='current'):
    with open(os.path.join(path, f'{dataset}.json'), 'r') as f:
        theory_file = json.load(f)

    # if its not a matrix, make it a matrix
    theory_error = np.sqrt(np.diag(theory_file[error_types[error_type]]))

    return theory_error

def create_comparison_plot(datasets, SM_projection_path, BSM_projection_path, theory_path, title, x_labels, plot_filename, y_label='value', modification_note='', add_ratios=True, label_rotation=0, x_font_size=16, y_font_size=16, title_font_size=20, dpi=300, error_type='current', log_scale=False):
    totalDataSM = np.array([])
    totalErrorSM = np.array([])
    totalDataBSM = np.array([])
    totalErrorBSM = np.array([])
    totalTheoryError = np.array([])

    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(hspace=0)

    for dataset in datasets:
        dataSM, errorSM = extract_data(SM_projection_path, dataset)
        dataBSM, errorBSM = extract_data(BSM_projection_path, dataset)
        theory_error = extract_theory(theory_path, dataset, error_type)

        # ----- Force 1D arrays -----
        dataSM = np.atleast_1d(dataSM).flatten()
        errorSM = np.atleast_1d(errorSM).flatten()
        dataBSM = np.atleast_1d(dataBSM).flatten()
        errorBSM = np.atleast_1d(errorBSM).flatten()
        theory_error = np.atleast_1d(theory_error).flatten()
        # cons = np.atleast_1d(cons).flatten()
        # agr = np.atleast_1d(agr).flatten()
        # --------------------------

        print(dataSM.shape, errorSM.shape, dataset)

        # if len(dataSM) != len(dataBSM):
        #     raise Exception('SM and BSM data lengths mismatch')

        # ids = np.array(list(range(len(dataSM))))

        # print(theory_error)
        # print(errorSM)
        totalDataSM = np.concatenate((totalDataSM, dataSM))
        totalErrorSM = np.concatenate((totalErrorSM, errorSM))
        totalDataBSM = np.concatenate((totalDataBSM, dataBSM))
        totalErrorBSM = np.concatenate((totalErrorBSM, errorBSM))
        totalTheoryError = np.concatenate((totalTheoryError, theory_error))

    ids = np.array(list(range(len(totalDataSM))))

    # plt.errorbar(ids, dataSM, yerr=errorSM, fmt='o', label=f'{dataset} SM', capsize=5)
    # plt.errorbar(ids, dataBSM, yerr=errorBSM, fmt='o', label=f'{dataset} BSM {modification_note}', capsize=5)
    # plt.ylabel(y_label, fontsize=16)
    # plt.xticks(ids, x_labels, rotation=0, ha='center', fontsize=16)
    # plt.title(title, fontsize=20)
    # plt.legend(fontsize=14)

    # plt.savefig(plot_filename)
    # plt.show()

    # fullSMerror = np.sqrt(errorSM**2+theory_error**2)
    # fullBSMerror = np.sqrt(errorBSM**2+theory_error**2)
    fullSMerror = np.sqrt(totalErrorSM**2+totalTheoryError**2)
    fullBSMerror = np.sqrt(totalErrorBSM**2+totalTheoryError**2)

    # plt.errorbar(ids, dataSM, yerr=fullSMerror, fmt='o', label=f'{dataset} SM', capsize=5)
    # plt.errorbar(ids, dataBSM, yerr=fullBSMerror, fmt='o', label=f'{dataset} BSM {modification_note}', capsize=5)
    # plt.ylabel(y_label, fontsize=16)
    # plt.xticks(ids, x_labels, rotation=0, ha='center', fontsize=16)
    # plt.title(title + ' (With theory error)', fontsize=16)
    # plt.legend(fontsize=14, loc='upper right')

    # plt.savefig(f'{plot_filename}_theory_err')
    # plt.show()
    shift = 0.04
    tiny_shift = 0.02
    # plt.errorbar(ids+shift+tiny_shift, dataSM, yerr=errorSM, fmt='o', color='blue', label=f'{dataset} SM', capsize=5)
    # plt.errorbar(ids+shift-tiny_shift, dataBSM, yerr=errorBSM, fmt='o', color='orange', label=f'{dataset} BSM {modification_note}', capsize=5)
    # plt.errorbar(ids-shift+tiny_shift, dataSM, yerr=fullSMerror, fmt='s', color='blue', label=f'{dataset} SM, + theory error', capsize=5)
    # plt.errorbar(ids-shift-tiny_shift, dataBSM, yerr=fullBSMerror, fmt='s', color='orange', label=f'{dataset} BSM {modification_note}, + theory error', capsize=5)
    if add_ratios:
        plt.subplot(2,1,1)
        ax_top = plt.gca()
    
    plt.errorbar(ids+shift, totalDataSM, yerr=totalErrorSM, fmt='o', color='blue', label=f'{dataset} SM', capsize=5)
    plt.errorbar(ids-shift, totalDataBSM, yerr=totalErrorBSM, fmt='o', color='orange', label=f'{dataset} BSM {modification_note}', capsize=5)
    plt.errorbar(ids+shift, totalDataSM, yerr=fullSMerror, fmt='o', color='blue', label=f'{dataset} SM, + theory error', capsize=5, alpha=0.25)
    plt.errorbar(ids-shift, totalDataBSM, yerr=fullBSMerror, fmt='o', color='orange', label=f'{dataset} BSM {modification_note}, + theory error', capsize=5, alpha=0.25)
    #plt.ylim(-10,25)
    plt.ylabel(y_label, fontsize=y_font_size)
    plt.yticks(fontsize=8)
    plt.xticks(ids, x_labels, rotation=label_rotation, ha='center', fontsize=x_font_size)
    plt.yscale('log' if log_scale else 'linear')
    plt.title(title, fontsize=title_font_size)
    plt.legend(fontsize=8)

    if add_ratios:
        plt.subplot(2,1,2)
        #ratio_SM = totalDataSM / totalDataSM
        ratio_BSM = totalDataBSM / totalDataSM
        #ratio_SM_error = fullSMerror / totalDataSM
        # Propagate the error for the ratio err a/b = (df/da)**2 * err_a**2 + (df/db)**2 * err_b**2, where df/da = 1/b and df/db = -a/b**2
        ratio_BSM_error = np.sqrt((totalErrorBSM / totalDataSM)**2 + (totalDataBSM * totalErrorSM / totalDataSM**2)**2)
        ratio_BSM_error_theory = np.sqrt((fullBSMerror / totalDataSM)**2 + (totalDataBSM * fullSMerror / totalDataSM**2)**2)

        #plt.errorbar(ids, ratio_SM, yerr=ratio_SM_error, fmt='o', color='blue', label=f'{dataset} SM', capsize=5)
        plt.errorbar(ids, ratio_BSM, yerr=ratio_BSM_error, fmt='o', color='orange', label=f'{dataset} BSM {modification_note}', capsize=5)
        plt.errorbar(ids, ratio_BSM, yerr=ratio_BSM_error_theory, fmt='o', color='orange', label=f'{dataset} BSM {modification_note} + theory error', capsize=5, alpha=0.25)
        plt.axhline(1, color='gray', linestyle='--')
        plt.ylabel('Ratio to SM', fontsize=16)
        # y tickmarks in scientific notation
        plt.yticks(fontsize=8)
        plt.xticks(ids, x_labels, rotation=label_rotation, ha='center', fontsize=x_font_size)
        #plt.title(title + ' (Ratio to SM)', fontsize=16)
        #plt.legend(fontsize=8)

        ax_bottom = plt.gca()
        ax_top.sharex(ax_bottom)
        ax_top.tick_params(axis='x', labelbottom=False)   # <-- REMOVE TOP LABELS

    plt.tight_layout()
    plt.savefig(plot_filename, bbox_inches='tight', dpi=dpi)
    plt.show()

if __name__ == "__main__":
    datasets = ['FCCee_Brw', 'FCCee_ww_161GeV', 'FCCee_ww_240GeV', 'FCCee_ww_365GeV']

    operator = '3pl2'

    BSM_path = f'/home/roan/smefit_bachelor_project/projections/injected_{operator}'
    SM_path = '/home/roan/smefit_bachelor_project/smefit_database/commondata_projections_L0'
    theory_path = '/home/roan/smefit_bachelor_project/smefit_database/theory'

    title = 'W leptonic branching ratios'
    branching_ratio_labels = [r'$B \left(W \to e \nu_e \right)$', r'$B \left(W \to \mu \nu_\mu \right)$', r'$B \left(W \to \tau \nu_\tau \right)$']
    plot_filename = f'BranchingRatios_c{operator}_001'
    note = f'c{operator} = 0.01'

    create_comparison_plot(datasets[0:1], SM_path, BSM_path, theory_path, 
                        title='W leptonic branching ratios', 
                        x_labels=branching_ratio_labels, 
                        plot_filename=f'BranchingRatios_c{operator}_001', 
                        modification_note=note)

    create_comparison_plot(datasets[1:4], SM_path, BSM_path, theory_path, 
                        title='W pair production cross sections', 
                        x_labels=[r'$\sigma$ at 161 GeV', r'$\sigma$ at 240 GeV', r'$\sigma$ at 365 GeV'],
                        plot_filename=f'WW_cross_sections_c{operator}_001', 
                        modification_note=note)
# create_comparison_plot(datasets[1], SM_path, BSM_path, theory_path, title, ['W pair cross section at 161GeV'], plot_filename, modification_note=note)
# create_comparison_plot(datasets[2], SM_path, BSM_path, theory_path, title, ['W pair cross section at 240GeV'], plot_filename, modification_note=note)
# create_comparison_plot(datasets[3], SM_path, BSM_path, theory_path, title, ['W pair cross section at 365GeV'], plot_filename, modification_note=note)

# groups = {
#     "GammaZ (GeV)": [0],
#     "SigmaHad (pb)": [1],
#     "R observables leptons": [2, 3, 4],
#     "R observables quarks" : [8, 9],
#     "A observables leptons": [5, 6, 7],
#     "A observables quarks" : [10, 11],
#     "alphaEW(mZ)": [12]
# }

# fig, axes = plt.subplots(len(groups), 1, figsize=(8, 10), sharex=False)

# for ax, (title, idxs) in zip(axes, groups.items()):
#     x = range(len(idxs))

#     d1 = [data1[i] for i in idxs]
#     e1 = [error1[i] for i in idxs]
#     d2 = [data2[i] for i in idxs]
#     e2 = [error2[i] for i in idxs]

#     ax.errorbar(x, d1, yerr=e1, fmt='o', label='W prime')
#     ax.errorbar(x, d2, yerr=e2, fmt='o', label='SM')

#     ax.set_title(title)
#     ax.set_xticks(x)
#     ax.set_xticklabels(idxs)

#     ax.legend()

# plt.tight_layout()
# plt.show()
