import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Digit Patterns (5 x 5)
# ==========================================

digits = {
    1: [0,1,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,1,1,1,0],

    2: [1,1,1,1,0,
        0,0,0,0,1,
        0,1,1,1,0,
        1,0,0,0,0,
        1,1,1,1,1],

    3: [1,1,1,1,0,
        0,0,0,0,1,
        0,1,1,1,0,
        0,0,0,0,1,
        1,1,1,1,0],

    4: [0,0,0,1,0,
        0,0,1,1,0,
        0,1,0,1,0,
        1,1,1,1,1,
        0,0,0,1,0],

    5: [1,1,1,1,1,
        1,0,0,0,0,
        1,1,1,1,0,
        0,0,0,0,1,
        1,1,1,1,0]
}


# ==========================================
# 2. Prepare Input and Target
# ==========================================

X = np.array(
    [digits[i] for i in range(1, 6)],
    dtype=float
)

# One-hot target
T = np.eye(5)


print("Input shape:", X.shape)
print("Target shape:", T.shape)


# ==========================================
# 3. Initialize Weights
# ==========================================

np.random.seed(42)

W = np.random.randn(25, 5) * 0.1

lr = 0.1
epochs = 1000

errors = []


# ==========================================
# 4. SGD Delta Learning Rule
# ==========================================

for epoch in range(epochs):

    total_error = 0

    for i in range(len(X)):

        # Calculate output
        y = np.dot(X[i], W)

        # Calculate error
        e = T[i] - y

        # SGD weight update
        W += lr * np.outer(X[i], e)

        # Calculate squared error
        total_error += np.sum(e ** 2)

    # Store error
    errors.append(0.5 * total_error)

    # Check convergence
    if total_error < 1e-4:

        print(f"Converged at epoch {epoch + 1}")
        break


print("\nTraining completed.")
print("Final Weight Shape:", W.shape)


# ==========================================
# 5. Convergence Curve
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(errors) + 1),
    errors,
    marker='o',
    markevery=50
)

plt.title("Convergence Curve — Digit Recognition")
plt.xlabel("Epoch")
plt.ylabel("Error")

plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================
# 6. Display Digits
# ==========================================

fig, axes = plt.subplots(
    1, 5,
    figsize=(10, 3)
)

for i, (digit, pattern) in enumerate(digits.items()):

    grid = np.array(pattern).reshape(5, 5)

    axes[i].imshow(
        grid,
        cmap='Greys',
        vmin=0,
        vmax=1
    )

    axes[i].set_xticks(
        np.arange(-0.5, 5, 1),
        minor=True
    )

    axes[i].set_yticks(
        np.arange(-0.5, 5, 1),
        minor=True
    )

    axes[i].grid(
        which='minor',
        color='black',
        linewidth=1.5
    )

    axes[i].tick_params(
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False
    )

    axes[i].set_title(f"Digit {digit}")


plt.suptitle("Five-by-Five Pixel Digits (1 to 5)")
plt.tight_layout()
plt.show()


# ==========================================
# 7. Digit Recognition / Testing
# ==========================================

print("\n" + "=" * 55)
print("Digit Recognition Results")
print("=" * 55)

for i in range(len(X)):

    # Calculate output
    output = np.dot(X[i], W)

    # Find neuron with maximum output
    predicted_class = np.argmax(output) + 1

    actual_class = i + 1

    print(
        f"Actual Digit: {actual_class} "
        f"| Predicted Digit: {predicted_class} "
        f"| Output: {np.round(output, 3)}"
    )

print("=" * 55)