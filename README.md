# 📉 Gradient Descent from Scratch — Linear Regression

A hands-on implementation of **Batch Gradient Descent** built from scratch in Python, benchmarked against scikit-learn's `LinearRegression`.

---

## 🧠 What This Project Does

This project demonstrates how **Gradient Descent** works under the hood for linear regression by:

- Implementing a custom `GDR` (Gradient Descent Regressor) class using only NumPy
- Training it on a synthetic regression dataset
- Comparing its R² score against scikit-learn's `LinearRegression`

---

## 📁 Project Structure

```
Gradient-Descent/
│
├── Gradient_Descent.ipynb   # Jupyter Notebook with step-by-step walkthrough
├── main.py                  # Clean Python script version
└── README.md
```

---

## ⚙️ How It Works

### Custom GDR Class

The `GDR` class manually computes the **partial derivatives** of Mean Squared Error (MSE) with respect to slope (`m`) and intercept (`b`), then updates them iteratively:

```
∂Loss/∂b = -2 * Σ(y - ŷ)
∂Loss/∂m = -2 * Σ((y - ŷ) * X)
```

Parameters are updated each epoch using:

```
b = b - lr * ∂Loss/∂b
m = m - lr * ∂Loss/∂m
```

---

## 📊 Results

| Model | R² Score |
|---|---|
| Custom GDR (50 epochs, lr=0.001) | ~0.6344 |
| Sklearn LinearRegression | ~0.6345 |

> ✅ Near-identical performance — validating the custom implementation!

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Run the script

```bash
python main.py
```

### Or open the notebook

```bash
jupyter notebook "Gradient_Descent.ipynb"
```

---

## 🔧 Hyperparameters

| Parameter | Value |
|---|---|
| Learning Rate | `0.001` |
| Epochs | `50` |
| Initial slope `m` | `100` |
| Initial intercept `b` | `-120` |

---

## 📦 Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

---

## 👨‍💻 Author

**Kaivalya Anil Patil**  
AI & Data Science Student — G H Raisoni College of Engineering & Management, Nagpur  
🔗 [LinkedIn](https://www.linkedin.com/in/kaivalya-anil-patil-67a52a305) | 🐙 [GitHub](https://github.com/Kaivalyp862)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
