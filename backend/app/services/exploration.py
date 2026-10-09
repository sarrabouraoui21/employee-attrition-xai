import json

import pandas as pd


def get_overview(
    df: pd.DataFrame,
    target: str
) -> dict:

    numeric = df.select_dtypes(include="number")
    missing = df.isna().sum()

    distribution = (
        df[target]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    return {
        "rows": len(df),
        "columns_count": len(df.columns),
        "columns": df.columns.tolist(),
        "types": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "missing_values": {
            col: int(count)
            for col, count in missing.items()
            if count > 0
        },
        "target_distribution": {
            str(key): int(value)
            for key, value in distribution.items()
        },
        "numeric_statistics": (
            json.loads(
        numeric.describe().round(2).T.to_json(
            orient="index"
        )
    )
    if len(numeric.columns) > 0
    else {}
)
    }


def get_feature_analysis(
    df: pd.DataFrame,
    column: str,
    target: str,
    positive_class: str
) -> dict:

    if column not in df.columns:
        raise ValueError(f"Variable inconnue : {column}")

    if column == target:
        raise ValueError(
            "Choisissez une variable explicative."
        )

    series = df[column]

    continuous = (
        pd.api.types.is_numeric_dtype(series)
        and series.nunique() > 12
    )

    if continuous:
        groups = pd.qcut(
            series,
            q=6,
            duplicates="drop"
        )
    else:
        groups = (
            series.astype("string")
            .fillna("Manquant")
        )

    analysis = pd.DataFrame({
        "group": groups,
        "positive": (
            df[target]
            .astype(str)
            .eq(positive_class)
            .astype(int)
        )
    })

    result = (
        analysis
        .groupby("group", observed=True, dropna=False)
        ["positive"]
        .agg(count="size", rate="mean")
        .reset_index()
    )

    return {
        "feature": column,
        "type": (
            "numeric" if continuous
            else "categorical"
        ),
        "groups": [
            {
                "label": str(row["group"]),
                "count": int(row["count"]),
                "positive_rate": round(
                    float(row["rate"]) * 100, 2
                )
            }
            for _, row in result.iterrows()
        ]
    }


def get_correlations(
    df: pd.DataFrame
) -> dict:

    numeric = df.select_dtypes(include="number")

    numeric = numeric.drop(
        columns=[
            "EmployeeNumber",
            "EmployeeCount",
            "StandardHours"
        ],
        errors="ignore"
    )

    numeric = numeric.loc[
        :, numeric.nunique() > 1
    ]

    corr = numeric.corr().round(2)

    return {
        "features": corr.columns.tolist(),
        "matrix": json.loads(
            corr.to_json(orient="values")
        )
    }