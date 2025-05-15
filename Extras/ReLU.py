import matplotlib.pyplot as plt
import numpy as np

# Generate 100 evenly spaced points between -10 and 10
x = np.linspace(-10, 10, 100)

# Compute the ReLU activation function for each point
y = np.maximum(0, x)

# Plot the ReLU activation function
plt.plot(x, y, label='f(x) = max(0, x)')

# Add a title to the plot
plt.title('ReLU Activation Function')

# Add label for the x-axis
plt.xlabel('x')

# Add label for the y-axis
plt.ylabel('max(0,x)')

# Enable grid for better readability
plt.grid(True)

# Add a legend to identify the function
plt.legend()

# Show the plot
plt.show()
