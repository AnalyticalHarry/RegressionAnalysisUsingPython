def histograms_with_distribution(df, columns, plots_per_row=4):
    num_plots = len(columns)
    nrows = int(np.ceil(num_plots / plots_per_row))
    fig, axs = plt.subplots(nrows=nrows, ncols=plots_per_row, figsize=(plots_per_row * 6, nrows * 4))
    axs = axs.flatten() 

    # Iterate over columns and create a histogram for each
    for i, column in enumerate(columns):
        if i < num_plots:
            # Plotting histogram
            sns.histplot(df[column], ax=axs[i], kde=True, color='black', alpha=0.8)

            # Calculate and annotate mean, median, and skewness
            mean_value = df[column].mean()
            median_value = df[column].median()
            skewness = df[column].skew()
            axs[i].annotate(f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\nSkewness: {skewness:.2f}', 
                            xy=(0.85, 0.9), xycoords='axes fraction', fontsize=10,
                            color='red', ha='right', va='center', bbox=dict(boxstyle='round', facecolor='white'))

            axs[i].set_title(f'Histogram of {column}')
            axs[i].grid(True, ls='--', color='grey', alpha=0.5)
    for i in range(num_plots, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

histograms_with_distribution(df, df.columns)