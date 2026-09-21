import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]], dtype=float)
D = np.array([0, 0, 1, 1], dtype=float)

W = np.zeros(3)
lr = 0.1
errors = []
for epoch in range(100):
    total_error = 0
    for i in range(len(X)):
        y = np.dot(W, X[i])
        e = D[i] - y
        W += lr * e * X[i]        # Delta rule
        total_error += e**2

    errors.append(0.5 * total_error)
    if total_error < 1e-4:
        print(f"Converged at epoch {epoch+1}")
        break

print("Final W:", W)
plt.plot(range(1, len(errors)+1), errors, marker='o')
plt.title("SGD Convergence (MSE vs Epoch)")
plt.xlabel("Epoch"); plt.ylabel("Error")
plt.grid(True); plt.show()
x1_vals = np.linspace(-0.5, 1.5, 100)
x2_vals = -(W[0]*x1_vals + W[2]) / W[1]

for i in range(len(X)):
    color = 'green' if D[i] == 1 else 'red'
    marker = '^' if D[i] == 1 else 'o'
    plt.scatter(X[i][0], X[i][1], color=color, marker=marker, s=150, zorder=5)
    plt.annotate(f"({int(X[i][0])},{int(X[i][1])})→{int(D[i])}",
                 (X[i][0]+0.03, X[i][1]+0.03))

plt.plot(x1_vals, x2_vals, 'k-', linewidth=2, label="Decision Boundary")
plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title("Decision Boundary — SGD Delta Rule")
plt.xlabel("x1"); plt.ylabel("x2")
plt.legend(); plt.grid(True); plt.show()
print(f"{'Input':>12} | {'Target':>6} | {'Output':>8} | {'Correct':>8}")
print("-" * 45)
for i in range(len(X)):
    out = np.dot(W, X[i])
    predicted = 1 if out >= 0.5 else 0
    correct = "Yes" if predicted == int(D[i]) else "No"
    print(f"{str(X[i][:2]):>12} | {int(D[i]):>6} | {out:>8.4f} | {correct:>8}")
