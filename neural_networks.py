import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle


result_dir = "results"
os.makedirs(result_dir, exist_ok=True)


# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
       
        # Store gradients for visualization
        self.gradients = None


    def activation(self, z):
        if self.activation_fn == 'tanh':
            return np.tanh(z)
        elif self.activation_fn == 'relu':
            return np.maximum(0, z)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-z))


    def activation_derivative(self, z):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(z)**2
        elif self.activation_fn == 'relu':
            return (z > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
       
    def forward(self, X):
        # Input to hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation(self.Z1)
       
        # Hidden to output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2))  # Sigmoid for binary classification
       
        return self.A2


    def backward(self, X, y):
        # Output layer error
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / X.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
       
        # Hidden layer error
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
       
        # Store gradients for visualization
        self.gradients = {'W1': dW1, 'W2': dW2}
       
        # Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y


# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
   
    # Perform a training step
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
   


    # Generate a grid in input space
    grid_size = 50
    x_range = np.linspace(-3, 3, grid_size)
    y_range = np.linspace(-3, 3, grid_size)
    xx_grid, yy_grid = np.meshgrid(x_range, y_range)
    grid_input = np.c_[xx_grid.ravel(), yy_grid.ravel()]


    # Map the grid through the network up to the hidden layer
    Z1_grid = np.dot(grid_input, mlp.W1) + mlp.b1
    A1_grid = mlp.activation(Z1_grid)
    Z2_grid = np.dot(A1_grid, mlp.W2) + mlp.b2
    A2_grid = 1 / (1 + np.exp(-Z2_grid))


    # Plot the grid in the hidden space
    ax_hidden.scatter(
        A1_grid[:, 0],
        A1_grid[:, 1],
        A1_grid[:, 2],
        c=A2_grid.ravel(),
        cmap='bwr',
        alpha=0.1
    )


    # Plot the hidden activations of the training data
    ax_hidden.scatter(
        mlp.A1[:, 0],
        mlp.A1[:, 1],
        mlp.A1[:, 2],
        c=y.ravel(),
        cmap='bwr',
        edgecolors='k',
        alpha=0.7
    )


    ax_hidden.set_xlabel('Neuron 1 Activation')
    ax_hidden.set_ylabel('Neuron 2 Activation')
    ax_hidden.set_zlabel('Neuron 3 Activation')
    ax_hidden.set_title(f"Hidden Space at Step {frame}")
   
    # Input space decision boundary
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)


    # Plot decision boundary
    ax_input.contourf(xx, yy, preds, levels=50, cmap='bwr', alpha=0.6)
    ax_input.contour(xx, yy, preds, levels=[0.5], colors='k', linewidths=2)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_xlabel('Feature 1')
    ax_input.set_ylabel('Feature 2')
    ax_input.set_title(f"Input Space at Step {frame}")
   
    # Positions of neurons in the network
    input_neurons = [(0, 0.5), (0, -0.5)]
    hidden_neurons = [(1, 1), (1, 0), (1, -1)]
    output_neuron = [(2, 0)]


    # Plot neurons and add labels
    # Input neurons
    input_labels = ['x1', 'x2']
    for idx, (x, y_pos) in enumerate(input_neurons):
        ax_gradient.scatter(x, y_pos, s=100, c='blue')
        ax_gradient.text(x - 0.1, y_pos + 0.1, input_labels[idx], fontsize=12, ha='right')


    # Hidden neurons
    hidden_labels = ['h1', 'h2', 'h3']
    for idx, (x, y_pos) in enumerate(hidden_neurons):
        ax_gradient.scatter(x, y_pos, s=100, c='green')
        ax_gradient.text(x, y_pos + 0.1, hidden_labels[idx], fontsize=12, ha='center')


    # Output neuron
    ax_gradient.scatter(output_neuron[0][0], output_neuron[0][1], s=100, c='red')
    ax_gradient.text(output_neuron[0][0] + 0.1, output_neuron[0][1], 'y', fontsize=12, ha='left')


    # Plot edges from input to hidden layer
    for i, (x1, y1) in enumerate(input_neurons):
        for j, (x2, y2) in enumerate(hidden_neurons):
            weight_grad = mlp.gradients['W1'][i, j]
            linewidth = np.clip(np.abs(weight_grad) * 100, 0.5, 5)
            linestyle = '-' if weight_grad >= 0 else '--'
            color = 'black'
            ax_gradient.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.7
            )


    # Plot edges from hidden to output layer
    for i, (x1, y1) in enumerate(hidden_neurons):
        x2, y2 = output_neuron[0]
        weight_grad = mlp.gradients['W2'][i, 0]
        linewidth = np.clip(np.abs(weight_grad) * 100, 0.5, 5)
        linestyle = '-' if weight_grad >= 0 else '--'
        color = 'black'
        ax_gradient.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=0.7
        )


    ax_gradient.axis('off')
    ax_gradient.set_title(f"Gradients at Step {frame}")


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)


    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)


    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)


    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()


if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
