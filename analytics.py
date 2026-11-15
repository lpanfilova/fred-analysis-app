import pandas as pd

def prepare_data(data):
    df = pd.DataFrame(
        data["observations"],
        columns=["date", "value"]
    )

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return (
        df.dropna()
          .sort_values("date")
          .set_index("date")["value"]
    )

def calculate_yoy(series, periods):
    return (series / series.shift(periods) - 1) * 100