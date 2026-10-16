import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_residuals(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))

    plt.title("Residual Error")
    plt.scatter(df["Predicted"], df["Error"], color="blue", marker="o")
    plt.axhline(y=0, color="black", linestyle="--")

    plt.xlabel("Predicted Value")
    plt.ylabel("Error (Actual - Predicted)")
    plt.grid(True)

    output_path = Path(__file__).resolve().parent.parent / "reports" / "residual_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

