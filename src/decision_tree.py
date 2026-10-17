"""Decision Tree (dt_over dan dt_under).   Jalankan: python -m src.decision_tree"""
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.experiment import run_model


def run() -> dict:
    return run_model(
        "dt",
        "7. DECISION TREE",
        lambda: DecisionTreeClassifier(**config.DT_PARAMS, random_state=config.RANDOM_STATE),
    )


if __name__ == "__main__":
    run()
