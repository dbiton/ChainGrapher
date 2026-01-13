import glob
import os

import matplotlib.ticker
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

from interfaces.interface import Interface

FIGS_DIR = './data/download/solana/figures'

figlatex=r'''\begin{figure}
\centerline{\includegraphics[width=\columnwidth]{figures/%s}}
\caption{%s}
\label{fig:%s}
\end{figure}'''
def maketuple(s:str):
    s1=s.replace('_','-')
    return (f"{s}.png", s1,s1)
def plot_graph(graph):
    plt.figure(figsize=(8, 6))
    pos = nx.kamada_kawai_layout(graph)  # positions for all nodes
    nx.draw(graph, pos, arrows=True)#, with_labels=True)
    plt.show()

def plot_block_size_distribution(df,print=print):
    bucket_width = 1
    block_sizes = list(df['txs'])
    
    # Determine the range of the data
    min_val = min(block_sizes)
    max_val = max(block_sizes)

    # Construct bin edges: start at min_val and go up to max_val in steps of 20
    # Adding a final 20 to max_val ensures we include the top edge
    bins = np.arange(min_val, max_val + bucket_width, bucket_width)
    
    weights = np.ones(len(block_sizes)) / len(block_sizes)
    
    # Plot the histogram
    plt.hist(block_sizes, bins=bins,weights=weights, edgecolor='black')

    # Add labels and title for clarity
    plt.xlabel('Block Size')
    plt.ylabel('Frequency')
    plt.xscale('log', base=2)
    # plt.grid()
    # plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{FIGS_DIR}/block_size_dist.png")
    plt.close()
    print(figlatex%maketuple("block_size_dist"))
def plot_density_distribution(df,print=print):
    bucket_width = 0.001
    block_sizes = list(df['density'])

    # Determine the range of the data
    min_val = min(block_sizes)
    max_val = max(block_sizes)

    # Construct bin edges: start at min_val and go up to max_val in steps of 20
    # Adding a final 20 to max_val ensures we include the top edge
    bins = np.arange(min_val, max_val + bucket_width, bucket_width)
    plt.figure(figsize=(20,4))
    weights = np.ones(len(block_sizes)) / len(block_sizes)

    # Plot the histogram
    counts, edges, bars = plt.hist(block_sizes, bins=bins, edgecolor='black')

    plt.bar_label(bars)
    # Add labels and title for clarity
    plt.xlabel('Density')
    plt.ylabel('Frequency')
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    ax=plt.gca()
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # plt.grid()
    # plt.legend()
    # plt.autoscale(enable=True, axis='x', tight=None)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{FIGS_DIR}/density_dist.png")
    plt.close()
    print(figlatex % maketuple("density_dist"))

def plot_smart_contract_percent(_df):
    plt.figure()
    
    quant_fill = 0.05
    bins_count = 32
    
    df = _df[_df["txs"] > 0].copy()
    block_number = df['Block Number']
    bins = np.linspace(block_number.min(), block_number.max(), num=bins_count)
    df['block_number_bin'] = pd.cut(block_number, bins=bins, include_lowest=True)
    prop = 'Value Transfer Ratio'
    df[prop] =  df['count_txs_value_transfer'] / df['txs']
    
    # Group by the bins and compute mean and SEM
    grouped = df.groupby('block_number_bin')
    mean_conflict = grouped["block_number"].mean()
    mean_prop = grouped[prop].mean()
    
    low_prop = grouped[prop].quantile(quant_fill)
    hi_prop = grouped[prop].quantile(1-quant_fill)

    # Plot mean density with confidence intervals
    plt.plot(mean_conflict, mean_prop)
    plt.fill_between(mean_conflict,
                    low_prop,
                    hi_prop,
                    alpha=0.2)
    
    # Add labels and title for clarity
    plt.xlabel('Block Number')
    plt.ylabel('Value-Transfer txs ratio')
    plt.grid()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{FIGS_DIR}/value_transfer_txs_ratio.png")
    plt.close()

def plot_call_metrics(_df):    
    quant_fill = 0.05
    bins_count = 64
    
    df = _df[_df["txs"] > 0].copy()
    block_number = df['block_number']
    bins = np.linspace(block_number.min(), block_number.max(), num=bins_count)
    df['block_number_bin'] = pd.cut(block_number, bins=bins, include_lowest=True)
    grouped = df.groupby('block_number_bin')
    mean_conflict = grouped["block_number"].mean()
    props = [
        "mean_call_count_smart_contract",
        "mean_call_height_smart_contract",
        "mean_call_count_leaves_smart_contract",
        "mean_call_degree_smart_contract"
    ]
    plt.figure()
    for prop in props: 
        mean_prop = grouped[prop].mean()
        
        low_prop = grouped[prop].quantile(quant_fill)
        hi_prop = grouped[prop].quantile(1-quant_fill)

        # Plot mean density with confidence intervals
        plt.plot(mean_conflict, mean_prop, label=prop)
        plt.fill_between(mean_conflict,
                        low_prop,
                        hi_prop,
                        alpha=0.2)
    
    # Add labels and title for clarity
    plt.xlabel('Block Number')
    plt.yscale('log', base=2)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{FIGS_DIR}/call_metrics.png")
    plt.close()

def plot_data_dir(csv_dir, chain_interface: Interface):
    all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    plot_data_filelist(all_files, chain_interface)

def plot_data_filelist(all_files, chain_interface: Interface):
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    with open(f"{FIGS_DIR}/figs.tex", 'a') as f:
       plot_data_df(df, chain_interface,print=lambda x:print(x,file=f))

def plot_data(csv_path, chain_interface: Interface):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset='block_number', keep='first')
    plot_data_df(df, chain_interface)
def plot_data_df(df, chain_interface: Interface,print=print):
    markers = ["o", "s", "^", "v", "D", "*"]
    
    lines_count = 4
    bins_count = 16
    quant_fill = 0.05


    additional_figs = chain_interface.get_additional_figures(df)
    for (fig, fig_name) in additional_figs:
        fig.savefig(os.path.join(FIGS_DIR, fig_name))
        plt.close(fig)
    
    # '''if is_callTracer:
    #     df['ratio_txs_value_transfer'] = df['count_txs_value_transfer'] / df['txs']
    #     plot_call_metrics(df)
    #     plot_smart_contract_percent(df)'''
    plot_block_size_distribution(df,print=print)
    plot_density_distribution(df,print=print)
    df['min_path_chromatic_ratio'] = df['longest_path_length_monte_carlo'] / df['greedy_color']
    df['max_path_chromatic_ratio'] = df['largest_conn_comp'] / df['clique_number']
    # df['user_tx_ratio'] = df['user_tx_count'] / df['txs']
    # df['system_tx_ratio'] = df['system_tx_count'] / df['txs']
    # df['mean_sui_transfered'] = df['total_sui_transfered'] / df['txs']
    
    # Ensure the data has X, Y, and other columns
    if "density" not in df.columns or "txs" not in df.columns:
        raise ValueError("The CSV file must contain 'X' and 'Y' columns.")

    # Extract X, Y, and property columns
    properties = df.drop(columns=["density"]).columns
    
    split_values = df["txs"].quantile([i / (lines_count) for i in range(1, lines_count)])
    split_values = sorted(list(split_values))
    # Create heatmaps for each property
    for prop in properties:
        if not pd.api.types.is_numeric_dtype(df[prop]):
            continue
        plt.figure()
        for i_tx_group in range(len(split_values) + 1):
            if i_tx_group == 0:
                txs_min = 0
                txs_max = split_values[i_tx_group]
            elif i_tx_group == len(split_values):
                txs_min = split_values[i_tx_group - 1]
                txs_max = float('inf')
            else:
                txs_min = split_values[i_tx_group - 1]
                txs_max = split_values[i_tx_group]

            # Select the data for the current tx_group
            df_group = df.loc[(df["txs"] <= txs_max) & (df["txs"] > txs_min)].sort_values(by=prop)
            
            # Copy to avoid SettingWithCopyWarning
            df_group = df_group.copy()
            
            density = df_group["density"]
            
            # Bin the 'prop' data
            if len(density) == 0:
                continue
            bins = np.linspace(density.min(), density.max(), num=bins_count)  # Adjust 'num' for bin granularity
            df_group['density_bin'] = pd.cut(density, bins=bins, include_lowest=True)
            
            # Group by the bins and compute mean and SEM
            grouped = df_group.groupby('density_bin', observed=False)
            mean_conflict = grouped["density"].mean()
            mean_prop = grouped[prop].mean()
            
            low_prop = grouped[prop].quantile(quant_fill)
            hi_prop = grouped[prop].quantile(1-quant_fill)

            # Plot mean density with confidence intervals
            plt.plot(mean_conflict, mean_prop, label=f"#txs>{int(txs_min)}", marker=markers[i_tx_group])
            plt.fill_between(mean_conflict,
                            low_prop,
                            hi_prop,
                            alpha=0.2)  # Adjust 'alpha' for transparency
            
        plt.grid()
        plt.ylabel(f"{prop}")
        plt.xlabel("Density")
        plt.legend()
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{FIGS_DIR}/{prop}.png")
        plt.close()
        print(figlatex % maketuple(prop))

    # print_overleaf_table(df)

def print_overleaf_table(df):
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Property & Mean & Std Dev & 5th Percentile & 95th Percentile \\")
    print(r"\hline")

    for prop in df.columns:
        if not pd.api.types.is_numeric_dtype(df[prop]):
            continue
        mean = df[prop].mean()
        std = df[prop].std()
        q05 = df[prop].quantile(0.05)
        q95 = df[prop].quantile(0.95)
        print(f"{prop} & {mean:.2f} & {std:.2f} & {q05:.2f} & {q95:.2f} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Summary statistics for each property}")
    print(r"\label{tab:summary_stats}")
    print(r"\end{table}")