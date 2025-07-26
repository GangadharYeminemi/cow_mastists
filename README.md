 Mastitis Detection Using Deep Learning + SVM

 Overview
A hybrid deep learning approach to detect **mastitis** in cows based on udder images. This project combines features from MViTv2 and ResNet50, followed by an SVM classifier.

 Methodology
- Dataset: Image classification (normal vs mastitis)
- Feature Extraction: MViTv2 & ResNet50 using `timm`
- Ensemble Learning: Concatenate features → PCA → SVM
- Evaluation: Accuracy comparison & classification reports
- Oversampling: SMOTE for class balance

Tech Stack
- PyTorch, Timm, SMOTE
- ResNet50, MViTv2, SVM, PCA, Matplotlib

Output
- Trained deep models: `*.pth`
- Trained SVM and PCA: `*.pkl`
- Plots: `accuracy_comparison.png`, `prediction_comparison.png`


