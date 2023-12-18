def boxplots_with_mean_median_outliers(df, columns, plots_per_row=4):
    num_plots = len(columns)
    nrows = int(np.ceil(num_plots / plots_per_row))
    fig, axs = plt.subplots(nrows=nrows, ncols=plots_per_row, figsize=(plots_per_row * 6, nrows * 4))
    axs = axs.flatten()  
    for i, column in enumerate(columns):
        if i < num_plots:
            sns.boxplot(x=df[column], ax=axs[i], color='black', flierprops=dict(markerfacecolor='black', marker='o'))
            mean_value = df[column].mean()
            median_value = df[column].median()
            axs[i].axvline(mean_value, color='red', lw=2, linestyle='--', label=f'Mean ({mean_value:.2f})')
            axs[i].axvline(median_value, color='cyan', linestyle=':', label=f'Median ({median_value:.2f})')
            axs[i].grid(True, ls='--', color='grey', alpha=0.5)
            axs[i].set_title(f'Box Plot of {column}')
            axs[i].legend()
    for i in range(num_plots, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

# boxplots_with_mean_median_outliers(df, df.columns)