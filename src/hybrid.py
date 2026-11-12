"""
src/hybrid.py
Hybrid of XGBoost and the MLP (blend or residual). XGBoost predictions used as inputs or targets MUST be out-of-fold by station (GroupKFold).
Run from the project root:  python -m src.hybrid
"""
