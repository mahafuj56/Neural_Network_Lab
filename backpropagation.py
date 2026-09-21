import numpy as np

# Inputs & Targets
x1, x2 = 0.05, 0.10
T1, T2 = 0.01, 0.99

# Initial weights
w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55

# Biases & Learning rate
b1, b2 = 0.35, 0.60
lr = 0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ---------- Forward Pass ----------
H1 = sigmoid(w1*x1 + w3*x2 + b1)
H2 = sigmoid(w2*x1 + w4*x2 + b1)

y1 = sigmoid(w5*H1 + w7*H2 + b2)
y2 = sigmoid(w6*H1 + w8*H2 + b2)

error = 0.5 * ((T1-y1)**2 + (T2-y2)**2)

print(f"H1={H1:.4f}, H2={H2:.4f}")
print(f"y1={y1:.4f}, y2={y2:.4f}")
print(f"Total Error={error:.6f}")


# ---------- Reset Weights ----------
w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55


# ---------- Backpropagation Training ----------
for epoch in range(10000):

    # Forward
    H1 = sigmoid(w1*x1 + w3*x2 + b1)
    H2 = sigmoid(w2*x1 + w4*x2 + b1)
    y1 = sigmoid(w5*H1 + w7*H2 + b2)
    y2 = sigmoid(w6*H1 + w8*H2 + b2)

    # Output deltas
    d1 = (y1-T1) * y1 * (1-y1)
    d2 = (y2-T2) * y2 * (1-y2)

    # Hidden deltas
    dH1 = (d1*w5 + d2*w6) * H1 * (1-H1)
    dH2 = (d1*w7 + d2*w8) * H2 * (1-H2)

    # Update output weights
    w5 -= lr*d1*H1
    w6 -= lr*d2*H1
    w7 -= lr*d1*H2
    w8 -= lr*d2*H2

    # Update input-hidden weights
    w1 -= lr*dH1*x1
    w2 -= lr*dH2*x1
    w3 -= lr*dH1*x2
    w4 -= lr*dH2*x2

    if epoch % 1000 == 0:
        loss = 0.5*((T1-y1)**2 + (T2-y2)**2)
        print(f"Epoch {epoch}: Loss={loss:.6f}, y1={y1:.4f}, y2={y2:.4f}")


# ---------- Final Result ----------
H1 = sigmoid(w1*x1 + w3*x2 + b1)
H2 = sigmoid(w2*x1 + w4*x2 + b1)
y1 = sigmoid(w5*H1 + w7*H2 + b2)
y2 = sigmoid(w6*H1 + w8*H2 + b2)

print("\nFinal Result:")
print(f"H1={H1:.4f}, H2={H2:.4f}")
print(f"y1={y1:.4f}, y2={y2:.4f}")
print(f"Final Error={0.5*((T1-y1)**2+(T2-y2)**2):.6f}")