import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Simulated loss landscape (parabola)
def loss_function(w):
    return w**2

# Simulating optimization steps for Gradient Descent and AdamW
w_gd, w_adamw = -2.5, -2.5  # Starting point
lr_gd, lr_adamw = 0.4, 0.4  # Learning rates

gd_steps, adamw_steps = [w_gd], [w_adamw]

# Optimizers (manual update for visualization)
optimizer_adamw = optim.AdamW([torch.tensor(w_adamw, requires_grad=True)], lr=lr_adamw)

for _ in range(10):
    # Gradient Descent update (manual)
    grad_gd = 2 * w_gd  # Derivative of x^2
    w_gd -= lr_gd * grad_gd
    gd_steps.append(w_gd)
    
    # AdamW update
    optimizer_adamw.zero_grad()
    loss = loss_function(optimizer_adamw.param_groups[0]['params'][0])
    loss.backward()
    optimizer_adamw.step()
    
    w_adamw = optimizer_adamw.param_groups[0]['params'][0].item()
    adamw_steps.append(w_adamw)

# Generate loss curve
x = np.linspace(-3, 3, 400)
y = x**2

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Loss Function", color="blue")

# Gradient Descent steps
plt.scatter(gd_steps, [loss_function(w) for w in gd_steps], color="red", label="Gradient Descent")
plt.plot(gd_steps, [loss_function(w) for w in gd_steps], linestyle="dashed", color="red", alpha=0.6)

# AdamW steps
plt.scatter(adamw_steps, [loss_function(w) for w in adamw_steps], color="green", label="AdamW")
plt.plot(adamw_steps, [loss_function(w) for w in adamw_steps], linestyle="dashed", color="green", alpha=0.6)

# Marking the optimal point
plt.scatter([0], [0], color="black", s=100, label="Optimal Point")

# Labels and title
plt.xlabel("Parameter Value")
plt.ylabel("Loss")
plt.title("Gradient Descent vs. AdamW Optimization")
plt.legend()
plt.grid()

# Show plot
plt.show()
