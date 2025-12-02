# Lab 4 – Knowledge Distillation Report  
Student ID：NM6141022

## 1. About Knowledge Distillation (15%)

### **Modes of Distillation used in this Lab**
本次實驗採用 **Offline Distillation** 模式。  
Teacher（ResNet50）會先以 60 epochs 完整訓練，並在蒸餾階段凍結參數；Student（ResNet18）僅利用 Teacher 的 logits 或 feature maps 進行學習，Teacher 不再更新。

### **Role of logits & effect of higher temperature**

Logits 是模型在 softmax 前的原始輸出，保留類別之間的相對差異，是知識蒸餾中「dark knowledge」的來源。  

KD 的 softmax with temperature：

$$
p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

較高的 Temperature 會：  
- 使分布變平滑（softer）  
- 放大 Teacher 對非正解類別的信心  
- 提供 Student 更多 inter-class 相似度資訊  

### **Which Teacher features are used (Feature-based KD)?**

在修改後的 `ResNet.forward()` 中，會回傳 residual blocks 的四層特徵：

```python
feature1 = self.layer1(x)
feature2 = self.layer2(feature1)
feature3 = self.layer3(feature2)
feature4 = self.layer4(feature3)
return out, [feature1, feature2, feature3, feature4]
```

即使用 **layer1, layer2, layer3, layer4** 的輸出作為蒸餾 feature maps。

---

## 2. Response-Based KD (30%)

### **How Temperature and α were chosen**

本次使用：

```python
T = 2.0
alpha = 0.7
```

原因：

- T=2 能讓 logits 分布變平滑，揭露 dark knowledge  
- α=0.7 代表學生模型 70% 學習老師、30% 學習 hard labels  

### **How the loss function was designed**

Response KD 的 loss 由兩部分構成。

---

### **(1) Distillation loss (Soft Target)**

對 logits 做 softmax(T) 後計算 KL divergence：

```python
student_log_soft = F.log_softmax(student_logits / T, dim=1)
teacher_soft = F.softmax(teacher_logits / T, dim=1)
kd_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (T * T)
```

對應公式：

$$
L_{KD} = T^2 \cdot KL(\text{softmax}(z_s/T),\ \text{softmax}(z_t/T))
$$

---

### **(2) Hard-label loss (CE)**

```python
ce_loss = F.cross_entropy(student_logits, targets)
```

---

### **(3) Final response-based KD Loss**

```python
loss = alpha * kd_loss + (1 - alpha) * ce_loss
```

公式：

$$
L = \alpha L_{KD} + (1 - \alpha) L_{CE}
$$

---

## 3. Feature-Based KD (30%)

### **How intermediate features were extracted**

在蒸餾時，同時取得 Teacher 與 Student 的四層 features：

```python
student_logits, student_features = self.student(x)
teacher_logits, teacher_features = self.teacher(x)
```

兩兩對應比較：

| Stage | Teacher shape | Student shape |
|-------|--------------|----------------|
| layer1 | 256 ch | 64 ch |
| layer2 | 512 ch | 128 ch |
| layer3 | 1024 ch | 256 ch |
| layer4 | 2048 ch | 512 ch |

因此需要 **1×1 conv connectors** 對齊維度。

---

## **How the feature-based loss function was designed**

### **(1) Channel alignment with 1×1 conv**

```python
self.connectors.append(
    nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
)
```

---

### **(2) Compute MSE loss across all 4 layers**

```python
def loss_fe(student_features, teacher_features, connectors):
    loss = 0.0
    mse = nn.MSELoss()
    for i in range(len(student_features)):
        student_f = connectors[i](student_features[i])
        teacher_f = teacher_features[i].detach()
        loss += mse(student_f, teacher_f)
    return loss / len(student_features)
```

公式：

$$
L_{feat} = \frac{1}{4} \sum_{i=1}^{4} \| g_i(f_s^{(i)}) - f_t^{(i)} \|_2^2
$$

---

### **(3) Combine feature loss with CE**

```python
loss = 0.3 * ce_loss + 0.7 * feature_loss
```

---

## 4. Comparison of Student Models (with & without KD) (5%)

| Model                     | loss | accuracy |
|--------------------------|------|----------|
| **Teacher from scratch** | 0.46 | **88.44%** |
| **Student from scratch** | 0.45 | **85.83%** |
| **Response-based student** | 0.82 | **85.12%** |
| **Featured-based student** | 0.55 | **86.69%** |

---

## 5. Implementation Observations and Analysis (20%)

### **Unexpected behavior**
Response-based KD 未優於 baseline（85.12% < 85.83%）。

---

### **Reasons**

1. **CIFAR-10 的 dark knowledge 不強**  
   Teacher 的分類邊界與 Student 類似，soft targets 帶來的額外資訊有限。

2. **Response KD 僅使用輸出層資訊**  
   缺乏中間層特徵的結構引導。

3. **Feature KD 更適合 ResNet50 → ResNet18**  
   深度差距大時，中間層特徵蒸餾效果更明顯，因此提升 +0.86%。

---

### **Difficulties and solutions**

| Issue | Solution |
|-------|----------|
| Feature channels 不一致 | 用 1×1 conv connectors 做 channel 對齊 |
| ResNet 預設不回傳 features | 改寫 forward 回傳 logits + 4 層特徵 |
| Teacher 不應更新 | 設定 teacher params：`requires_grad=False` |
| Loss 組合比例需調整 | 使用 β=0.3 加強 feature loss |

