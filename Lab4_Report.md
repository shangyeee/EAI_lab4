# Lab 4 – Knowledge Distillation Report  
Student ID：NM6141022

## 1. About Knowledge Distillation (15%)

### **Modes of Distillation used in this Lab**
本次實驗採用 **Offline Distillation** 模式。  
Teacher（ResNet50）會先以 60 epochs 完整訓練，並在蒸餾階段凍結參數；Student（ResNet18）僅利用 Teacher 的 logits 或 feature maps 進行學習，Teacher 不再更新。

### **Role of logits & effect of higher temperature**
Logits 是模型在 softmax 前的原始輸出，保留類別之間的相對差異，是知識蒸餾中「dark knowledge」的來源。  
KD 中會使用溫度 Softmax：

\[
p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
\]

較高的 Temperature 會：  
- 使分布變平滑（softer）；  
- 放大 Teacher 在非正解類別上的細微信心；  
- 提供 Student 更多 inter-class similarity 資訊。

### **Which Teacher features are used (Feature-based KD)?**
在修改後的 `ResNet.forward()` 中，我從以下四個 residual stages 擷取中間特徵：

```python
feature1 = self.layer1(x)
feature2 = self.layer2(feature1)
feature3 = self.layer3(feature2)
feature4 = self.layer4(feature3)
return out, [feature1, feature2, feature3, feature4]
```

即使用 **layer1, layer2, layer3, layer4** 的輸出作為蒸餾用的 feature maps。

---

## 2. Response-Based KD (30%)

### **How Temperature and α were chosen**
我採用：

```python
T = 2.0
alpha = 0.7
```

選擇原因：

- T=2 能平滑 logits 分布，揭露 Teacher 的 dark knowledge，同時不會過度平均；
- α=0.7 表示較重視 Teacher 的 soft targets（70%），但保留 30% 的 ground-truth supervision。

### **How the loss function was designed**

Response KD 的 loss 由兩部分組成：

#### **(1) Distillation (soft target) loss**
對 logits 進行溫度 Softmax，再做 KL divergence：

```python
student_log_soft = F.log_softmax(student_logits / T, dim=1)
teacher_soft = F.softmax(teacher_logits / T, dim=1)
kd_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (T * T)
```

公式：

\[
L_{KD} = T^2 \cdot KL(\text{softmax}(z_s / T),\ \text{softmax}(z_t / T))
\]

#### **(2) Hard-label (CE) loss**

```python
ce_loss = F.cross_entropy(student_logits, targets)
```

#### **(3) Final Response KD Loss**

```python
loss = alpha * kd_loss + (1 - alpha) * ce_loss
```

\[
L = \alpha L_{KD} + (1 - \alpha)L_{CE}
\]

---

## 3. Feature-Based KD (30%)

### **How intermediate features were extracted**
在 Student 與 Teacher 的 forward pass 中，都會回傳四組 residual stage features：

```python
student_logits, student_features = self.student(x)
teacher_logits, teacher_features = self.teacher(x)
```

蒸餾時會兩兩對應比較：

| Stage | Teacher shape | Student shape |
|-------|----------------|----------------|
| layer1 | 256 channels | 64 channels |
| layer2 | 512 channels | 128 channels |
| layer3 | 1024 channels | 256 channels |
| layer4 | 2048 channels | 512 channels |

因此需要 connector 對齊維度。

---

### **How the feature-based loss function was designed**

#### **(1) Aligning channels using 1×1 conv connectors**

```python
self.connectors.append(
    nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
)
```

#### **(2) Compute MSE across all feature stages**

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

數學式：

\[
L_{\text{feat}} = \frac{1}{4}\sum_{i=1}^4 ||g_i(f_s^{(i)}) - f_t^{(i)}||_2^2
\]

#### **(3) Combine feature loss with CE loss**
本次使用 β=0.3（較重視 feature 学习）：

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
Response-based KD 出現了意外現象：結果 **沒有優於 baseline Student**，反而略微下降（85.12% < 85.83%）。

### **Reasons**
1. **CIFAR-10 的 logits dark knowledge 不強**  
   Teacher 與 Student 在此資料集的 decision boundary 高度相似，因此 Teacher 的 soft targets 帶來的額外訊息差距有限。

2. **Response KD 僅作用於輸出層**  
   中階特徵往往更能補足表示能力，因此 output-only distillation 效果有限。

3. **Feature KD 更適合 ResNet50 → ResNet18**  
   在深度差距大的 teacher–student 配對中，特徵蒸餾能給 Student 更完整的結構訊息，因此取得 +0.86% 的提升。

### **Difficulties and solutions**
| Issue | Solution |
|-------|----------|
| Feature channels 不一致 | 使用 1×1 conv connectors 做 channel 對齊 |
| ResNet 無法同時回傳 features | 修改 forward 回傳 logits + 4 層特徵 |
| Teacher 不應更新 | 對 Teacher 設定 `requires_grad=False` |
| Loss 組合比例調整 | 使用 β=0.3 強化 feature loss 影響力 |
