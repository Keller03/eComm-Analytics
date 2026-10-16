from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def train_regression(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=69)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

    return model, Y_test, predictions

def evaluate_regression(model, Y_test, predictions) -> dict:
    return {
        "metrics": {
            "R2": r2_score(Y_test, predictions),
            "MAE": mean_absolute_error(Y_test, predictions),
            "MSE": mean_squared_error(Y_test, predictions)
        },
        "parameters": {
            "intercept": model.intercept_,
            "coefficients": dict(zip(model.feature_names_in_, model.coef_))
        }
    }

def print_evaluation(evaluation: dict) -> None:
    for category, values in evaluation.items():
        print(f"\n--- {category.upper()} ---")

        for key, value in values.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for feature, coefficient in value.items():
                    print(f"  {feature}: {coefficient:.3f}")
            else:
                print(f"{key}: {value:.3f}")

def calculate_residuals(Y_test, predictions) -> pd.DataFrame:
    results = pd.DataFrame({
        "Actual": Y_test,
        "Predicted": predictions
    })

    results["Error"] = results["Actual"] - results["Predicted"]
    results["Absolute Error"] = results["Error"].abs()

    return results

def print_residuals(residuals: pd.DataFrame) -> None:
    print(f"\n--- TOP 10 ABSOLUTE ERRORS ---")
    print(residuals.sort_values("Absolute Error", ascending=False).head(10))

    print(f"\n--- MEAN ERROR ---")
    print(residuals["Error"].mean())