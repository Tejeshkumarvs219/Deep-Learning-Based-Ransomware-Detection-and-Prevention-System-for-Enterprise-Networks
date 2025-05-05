# Deep-Learning-Based-Ransomware-Detection-and-Prevention-System-for-Enterprise-Networks
Deep Learning-Based Ransomware Detection System using CNN, LSTM, and Hybrid Models trained on the RanSAP dataset. Achieved over 81% accuracy in multi-class classification across 26 ransomware families. Feature engineering, model comparisons, and cross-validation included.

# 🛡️ Ransomware Detection using Deep Learning

This project explores deep learning-based detection of ransomware families using CNN, LSTM, and hybrid CNN-LSTM models. The work was done as part of my Master’s Project at UMass Dartmouth.

## 🔍 Overview
- Dataset: **RanSAP** (26 ransomware families)
- Models: **CNN, FCNN, LSTM, Hybrid CNN-LSTM**
- Key Metrics: Hybrid model achieved **81.17% accuracy**
- Tools: PyTorch, NumPy, scikit-learn, pandas

## 📁 Structure
- `models/`: Models
- `documentation/`: Report
- `results/`: Confusion matrix and visualizations
- `codes/`: Python source code files

## 🧠 Feature Engineering
- Entropy & write amplification factor
- Normalization & class balancing
- Missing value handling

## 📊 Cross Validation
Used 3-Fold Stratified CV on CNN and LSTM

---

This project demonstrates the power of deep learning in malware forensics and threat detection.
