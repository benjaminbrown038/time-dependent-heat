# main.py
import os
from heat import solve_heat
from dataset import build_dataset
from models import MLP
from train import train_model
from plots import plot_slice

def main():
    cfg = {
        "nx": 30, "ny": 30, "nz": 30,
        "steps": 500, "dt": 1e-3,
        "hidden": 128, "depth": 3,
        "lr": 1e-3, "epochs": 600,
        "outdir": "experiments/results/heat_time"
    }

    os.makedirs(cfg["outdir"], exist_ok=True)

    # PHYSICS SOLVER
    u = solve_heat(cfg)

    # DATASET (x,y,z → temperature)
    pts, vals = build_dataset(u)

    # NN SURROGATE
    model = MLP(cfg["hidden"], cfg["depth"])
    train_model(model, pts, vals, cfg)

    # VISUALIZE RESULT
    plot_slice(u, cfg["outdir"])

if __name__ == "__main__":
    main()
