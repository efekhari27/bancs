#
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#
from matplotlib import rc
rc('font', **{'family': 'Libertine'})
rc('text', usetex=True)
rc('font', size=12)# Set the default text font size
rc('axes', titlesize=16)# Set the axes title font size
rc('axes', labelsize=12)# Set the axes labels font size
rc('xtick', labelsize=12)# Set the font size for x tick labels
rc('ytick', labelsize=12)# Set the font size for y tick labels
rc('legend', fontsize=12)# Set the legend font size

#
def draw_mean(df, save_file=None):
    fig = plt.figure(figsize=(5, 4))
    mark = ['x', '.', 'd']
    #mark = ['$-$', '.', '$/$']
    for i, method in enumerate(df["method"].unique()):
        sdf = df[df["method"]==method].copy()
        #
        plt.plot(sdf["nb_samples"], sdf['pf_mean'], marker=mark[i], color=f"C{i}", label=method)
        # plt.fill_between(sdf["nb_samples"], sdf['pf_ic_low'], sdf['pf_ic_up'], color=f"C{i}", alpha=0.2, label="CI 95\%")
    plt.axhline(df["pf_ref"].unique(), color="k", zorder=0)
    plt.legend(ncols=3, bbox_to_anchor=(1.1, -0.17))
    plt.xlabel("Total sample size ($N \\times k_\#$)")
    plt.ylabel("Average $\widehat{p_\mathrm{f}}$ ($10^3$ rep.)")
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    return fig

def draw_cov(df, save_file=None):
    fig = plt.figure(figsize=(5, 4))
    mark = ['x', '.', 'd']
    #mark = ['$-$', '.', '$/$']
    for i, method in enumerate(df["method"].unique()):
        sdf = df[df["method"]==method].copy()

        #
        plt.plot(sdf["nb_samples"], sdf['pf_std'] / sdf['pf_mean'], marker=mark[i], color=f"C{i}", label=method)
    plt.legend(ncols=3, bbox_to_anchor=(1.1, -0.17))
    plt.xlabel("Total sample size ($N \\times k_\#$)")
    plt.ylabel("Average $\widehat{\mathrm{cov}}$ ($10^3$ rep.)")
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    return fig


for ebc_tuning in ["AMISE", "LogLikelihood", "PenalizedKL"]:
    ## Four Branch problem
    df = pd.read_csv(f"bancs_results/RP4B_results_{ebc_tuning}.csv")
    draw_mean(df, save_file=f"bancs_figures/RP4B_mean_{ebc_tuning}.png")
    draw_cov(df, save_file=f"bancs_figures/RP4B_cov_{ebc_tuning}.png")

    ## Reliability problem 38
    df = pd.read_csv(f"bancs_results/RP38_results_{ebc_tuning}.csv")
    draw_mean(df, save_file=f"bancs_figures/RP38_mean_{ebc_tuning}.png")
    draw_cov(df, save_file=f"bancs_figures/RP38_cov_{ebc_tuning}.png")

    # ## Oscillator case
    # df = pd.read_csv(f"bancs_results/Oscillator_results_{ebc_tuning}.csv")
    # draw_mean(df)
    # plt.legend(ncols=2, bbox_to_anchor=(0.95, -0.17))
    # plt.savefig(f"bancs_figures/Oscillator_mean_{ebc_tuning}.png", dpi=300, bbox_inches='tight')
    # draw_cov(df)
    # plt.legend(ncols=2, bbox_to_anchor=(0.95, -0.17))
    # plt.savefig(f"bancs_figures/Oscillator_cov_{ebc_tuning}.png", dpi=300, bbox_inches='tight');
