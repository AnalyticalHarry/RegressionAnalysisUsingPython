import matplotlib.pyplot as plt
import math

def scatter_plots(x, y_list, df, n_rows=3):
    #number of plots and ensure a minimum of 3 rows
    num_plots = len(y_list)
    n_rows = max(n_rows, 3)
    #number of rows needed
    rows = math.ceil(num_plots / n_rows)  
    #figure and a grid of subplots
    fig, axes = plt.subplots(nrows=rows, ncols=n_rows, figsize=(16, 4 * rows))
    #grid of subplots into a 1D array
    axes_flat = axes.flatten()  
    axes_dict = {}  
    for i, y in enumerate(y_list):
        ax = axes_flat[i]  
        ax.scatter(df[x], df[y], color='black', s=20)  
        ax.grid(True, ls='--', alpha=0.3, color='black') 
        ax.set_title(f'{x} vs {y}') 
        ax.set_ylabel(y)  
        ax.set_xlabel(x) 
        axes_dict[f'ax{i+1}'] = ax  
    for ax in axes_flat[num_plots:]:
        ax.set_visible(False)
    plt.tight_layout()  
    plt.show()  


#scatter_plots('bodyfat', ['density', 'age', 'weight', 'height', 'neck', 'chest', 'abdomen', 'hip', 'thigh', 'knee', 'ankle', 'biceps', 'forearm','wrist'], df, n_rows=4)