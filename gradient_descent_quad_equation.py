def gradient_descent(ax, learning_rate):
    #gradient descent
    final_x, history = gradient_descent_with_history(gradient, initial_x, learning_rate, iterations, tolerance)

    ax.plot(x_values, y_values, label=f'Function: $f(x) = x^2 - 4x + 1$', lw=2)
    ax.scatter(history, [func(x) for x in history], color='red', zorder=5, label='GD Steps')
    ax.plot(history, [func(x) for x in history], 'r--', lw=1, zorder=5)
    ax.annotate('Start', xy=(history[0], func(history[0])), xytext=(history[0], func(history[0]) + 10),
                arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
    ax.annotate('End', xy=(final_x, func(final_x)), xytext=(final_x, func(final_x) + 10),
                arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
    ax.set_title(f'Learning Rate: {learning_rate}')
    ax.set_xlabel('x value')
    ax.set_ylabel('f(x)')
    ax.grid(True, alpha=0.5, ls='--', color='grey')
    ax.legend()

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

#x values for the function and their corresponding y values
x_values = np.linspace(-2, 11, 400)
y_values = func(x_values)

#parameters
initial_x = 9
iterations = 100
tolerance = 1e-6
learning_rates = [0.1, 0.3, 0.8, 0.9]

#plot learning rate
for ax, lr in zip(axs.flatten(), learning_rates):
    gradient_descent(ax, lr)
plt.tight_layout()
plt.show()
