import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 실제 baseline 값 (사용자가 제공한 값)
mu, sigma = 0.0628, 0.0543
example_scores = [0.4123, 0.4769, 0.6015, 0.6851, 0.6364, 0.6291, 0.5161, 0.3229]  

# 정규분포 곡선
x = np.linspace(-0.1, 1.0, 500)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p,label=f"Mean :μ={mu:.2f}")
plt.axvline(mu, color="black", linestyle="--", linewidth=2, label=f"Mean={mu:.2f}")


# 실제 유사쌍 위치
for s in example_scores:
    plt.axvline(s, color="orange", linestyle="-", linewidth=2)

plt.xlabel("Cosine similarity")
plt.xticks(fontsize=10)
plt.yticks([])
plt.legend()
plt.show()
