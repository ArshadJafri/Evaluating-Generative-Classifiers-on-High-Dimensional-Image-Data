# 🧠 | RGB vs Grayscale in Generative Models

This project evaluates the impact of input dimensionality on the performance of generative classifiers: **Gaussian Naive Bayes**, **Linear Discriminant Analysis (LDA)**, and **Quadratic Discriminant Analysis (QDA)**.

## 🎯 Objective

To analyze how models perform when trained on high-dimensional RGB images versus their grayscale counterparts, focusing on classification accuracy and parameter estimation complexity.

## 📊 Models and Results

| Model  | RGB Accuracy | Grayscale Accuracy |
|--------|--------------|--------------------|
| GNB    | (Not stated) | (Not stated)       |
| LDA    | (Not stated) | (Not stated)       |
| QDA    | **36.23%**   | **44.28%**         |

## 🧪 Key Insights

- **QDA** struggles with RGB inputs due to high parameter count and overfitting risks.
- **Grayscale images** reduce feature dimensionality (from 3072 to 1024), enabling more stable model training and higher test accuracy.
- **LDA** benefits from shared covariance assumptions, leading to lower complexity compared to QDA.

## 📌 Summary

This project highlights the importance of feature dimensionality when training generative classifiers on image data. Grayscale inputs simplify computation and enhance performance in resource-constrained or small-data settings.

---

Created by Arshad Jafri Shaik Mohammed | UTA ID: 1002197716
