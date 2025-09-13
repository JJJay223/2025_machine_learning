# 1.Read《Deep Learning: An Introduction for Applied Mathematicians》Consider a network as defined in (3.1) and (3.2). Assume that we want to compute $\nabla a^{[L]}(x)$, find an algorithm to calculate it ($n_L=1$).


---

## Define

(Feedforward Neural Network) in (3.1) and (3.2)：

- **輸入層 [0]**：
  $$a^{[0]} = x$$

- **逐層計算 (l = 1, 2, ..., L)**：
  $$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]} \quad (3.1)$$
  
  $$a^{[l]} = \sigma(z^{[l]}) \quad (3.2)$$

其中：
- $L$：總層數。
- $x$：輸入向量。
- $a^{[l]}$：第 $l$ 層的激活向量（輸出）。
- $W^{[l]}$：第 $l$ 層的權重矩陣。
- $b^{[l]}$：第 $l$ 層的偏置向量。
- $z^{[l]}$：第 $l$ 層的加權輸入。
- $\sigma$：非線性激活函數。

---

## (Backpropagation)

### 1. 前向傳播 (Forward Pass)

輸入 $x$ ，依序計算並儲存每一層的加權輸入 $z^{[l]}$ 和激活值 $a^{[l]}$，直到最終輸出 $a^{[L]}$。

1.  **初始化**：設定 $a^{[0]} = x$。
2.  **迭代計算**：對於 $l = 1, 2, \dots, L$，依序計算並儲存：
    $$
    z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}
    $$
    $$
    a^{[l]} = \sigma(z^{[l]})
    $$

這些儲存下來的中間值會在反向傳播階段被使用。

### 2. 反向傳播 (Backward Pass)

從最後一層開始，反向、逐層地計算梯度。 定義 $g^{[l]} = \nabla_{a^{[l]}} a^{[L]}$，為最終輸出 $a^{[L]}$ 對第 $l$ 層激活值 $a^{[l]}$ 的梯度。 最終求得 $g^{[0]}$，因為 $a^{[0]} = x$。

1.  **初始化 (在第 L-1 層)**：計算輸出 $a^{[L]}$ 對其前一層激活值 $a^{[L-1]}$ 的梯度。
    $$
    g^{[L-1]} = \nabla_{a^{[L-1]}} a^{[L]} = (W^{[L]})^T \sigma'(z^{[L]})
    $$
    *因為 $a^{[L]}$ 是純量（題目假設 $n_L=1$），所以 $\sigma'(z^{[L]})$ 也是一個純量。*

2.  **反向迭代**：對於 $l = L-1, L-2, \dots, 1$，利用 $g^{[l]}$ 計算 $g^{[l-1]}$：
    $$
    g^{[l-1]} = (W^{[l]})^T (g^{[l]} \odot \sigma'(z^{[l]}))
    $$
    其中 $\odot$ 代表 Hadamard Product。

3.  **最終結果**：當迭代完成後，得到 $g^{[0]}$，即為所求的梯度：
    $$
    \nabla_{x} a^{[L]}(x) = g^{[0]}
    $$

# 2.There are unanswered questions during the lecture, and there are likely more questions we haven't covered. Take a moment to think about them and write them down here.

1.  對於二元分類問題，loss function好像更常使用Cross-Entropy，MSE在（one-hot encoder)下

     假如使用sigmoid：σ(z)的導數是σ(z)(1-σ(z))
     真實為１預測輸出非常錯誤 假設為0.01時
     
     就會導致梯度變得非常小，幾乎為零

     因為MSE 假設預測目標是連續的，且誤差服從高斯分佈。
     但分類問題是離散的，One-hot encoder雖然轉化為數值向量，但不知道理論上這樣是不是合理的？
2.  Approximation Theory提到可以用 p 個神經元來近似 $x^{2p-1}$，如果要近似更高次的多項式，是否需要更多神經元的網路，和網路深度是否效果不同？

3.  LWLR需要透過每一個預測點x計算權重擬合新模型，計算成本應該很高？
