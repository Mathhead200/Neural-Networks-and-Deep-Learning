# k-NN MNIST experiment

This small project trains k-NN classifiers on MNIST CSV files and measures accuracy vs prediction time.

Files:
- `a.py` - main script. Run with `python a.py`.
- `mnist_train.csv`, `mnist_test.csv` - expected CSV datasets (labels in first column or column named `label`).
- `requirements.txt` - Python dependencies.

Quick start (Windows cmd):

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python a.py
```

Examples:
- Run with a range: `python a.py --ks 1-15:2`
- Subsample training to 5000 examples to speed up: `python a.py --train-sample 5000`
