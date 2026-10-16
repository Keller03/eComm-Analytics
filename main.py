from src.data_loader import loadData
from src.analysis import inspect_data, print_inspection, calculate_pearson_correlation
from src.regression import (
    train_regression,
    evaluate_regression,
    print_evaluation,
    calculate_residuals,
    print_residuals
)
from src.visualization import plot_residuals

from pathlib import Path


def main():
    df = loadData(Path(__file__).parent / "data" / "ecommerce.csv")

    analysis = inspect_data(df)
    print_inspection(analysis)

    correlations = calculate_pearson_correlation(df)
    print("\n--- CORRELATIONS ---")
    print(correlations.round(3))

    X = df[[
        "Avg. Session Length",
        "Time on App",
        "Time on Website",
        "Length of Membership"
    ]]
    Y = df["Yearly Amount Spent"]

    model, Y_test, predictions = train_regression(X, Y)

    evaluation = evaluate_regression(model, Y_test, predictions)
    print_evaluation(evaluation)

    residuals = calculate_residuals(Y_test, predictions)
    print_residuals(residuals)
    plot_residuals(residuals)


if __name__ == "__main__":
    main()

