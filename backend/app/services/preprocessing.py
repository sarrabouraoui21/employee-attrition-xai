
import json
import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.schemas.preprocessing import PreprocessingConfig
from app.services.datasets import DatasetContext

logger = logging.getLogger(__name__)

KAGGLE_EXCLUSIONS = {
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18"
}


@dataclass
class PreparedData:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    numeric_features: list[str]
    categorical_features: list[str]


# =====================================================
# 1. AVAILABLE FEATURES
# =====================================================

def available_features(context: DatasetContext) -> list[str]:
    df = context.df

    excluded = {context.target}

    if context.source == "kaggle":
        excluded.update(KAGGLE_EXCLUSIONS)

    return [
        col for col in df.columns
        if col not in excluded
        and df[col].nunique(dropna=True) > 1
    ]


def get_preprocessing_options(context: DatasetContext) -> dict:
    features = available_features(context)

    numeric = [
        col for col in features
        if pd.api.types.is_numeric_dtype(context.df[col])
    ]

    categorical = [
        col for col in features
        if col not in numeric
    ]

    return {
        "dataset_id": context.dataset_id,
        "target": context.target,
        "positive_class": context.positive_class,
        "negative_class": context.negative_class,
        "available_features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "excluded_features": [
            col for col in context.df.columns
            if col not in features
            and col != context.target
        ]
    }


# =====================================================
# 2. PREPROCESSING PIPELINE
# =====================================================

def prepare_dataset(
    context: DatasetContext,
    config: PreprocessingConfig
) -> PreparedData:

    df = context.df

    logger.info(
        "PREPROCESSING START | dataset=%s | target=%s",
        context.dataset_id,
        context.target
    )

    allowed = available_features(context)

    features = (
        allowed if config.features is None
        else config.features
    )

    if not features:
        raise ValueError(
            "Sélectionnez au moins une variable explicative."
        )

    if len(features) != len(set(features)):
        raise ValueError("Variables dupliquées.")

    invalid = set(features) - set(allowed)

    if invalid:
        raise ValueError(
            f"Variables interdites ou inconnues : {sorted(invalid)}"
        )

    X = df[features].copy()

    # La classe positive correspond à 1.
    y = (
        df[context.target]
        .astype(str)
        .eq(context.positive_class)
        .astype(int)
    )

    if y.nunique() != 2:
        raise ValueError(
            "La cible doit contenir deux classes."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Il faut au moins deux observations par classe."
        )

    test_rows = math.ceil(len(df) * config.test_size)

    if test_rows < 2 or len(df) - test_rows < 2:
        raise ValueError(
            "Dataset trop petit pour un split stratifié. "
            "Augmentez la proportion du test."
        )

    # -------------------------------------------------
    # A. TRAIN / TEST SPLIT
    # -------------------------------------------------

    logger.info(
        "STEP 1 | Train/test split | test_size=%.2f",
        config.test_size
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y
        )
    except ValueError as exc:
        raise ValueError(
            f"Split stratifié impossible : {exc}"
        ) from exc

    numeric_features = (
        X_train.select_dtypes(include="number")
        .columns.tolist()
    )

    categorical_features = [
        col for col in features
        if col not in numeric_features
    ]

    empty_columns = [
        col for col in features
        if X_train[col].isna().all()
    ]

    if empty_columns:
        raise ValueError(
            "Variables entièrement manquantes dans le train : "
            f"{empty_columns}"
        )

    logger.info(
        "STEP 1 OK | train=%d | test=%d",
        len(X_train),
        len(X_test)
    )

    # -------------------------------------------------
    # B. NUMERIC IMPUTATION + SCALING
    # -------------------------------------------------

    transformers = []

    if numeric_features:
        numeric_steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy=config.numeric_imputation
                )
            )
        ]

        if config.scale_numeric:
            numeric_steps.append(
                ("scaler", StandardScaler())
            )

        transformers.append(
            (
                "num",
                Pipeline(numeric_steps),
                numeric_features
            )
        )

    logger.info(
        "STEP 2 | Numerical imputation=%s | scaling=%s",
        config.numeric_imputation,
        config.scale_numeric
    )

    # -------------------------------------------------
    # C. CATEGORICAL IMPUTATION + ENCODING
    # -------------------------------------------------

    if categorical_features:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ])

        transformers.append(
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        )

    logger.info(
        "STEP 3 | OneHotEncoding | numeric=%d | categorical=%d",
        len(numeric_features),
        len(categorical_features)
    )

    # -------------------------------------------------
    # D. FIT ON TRAIN ONLY
    # -------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True
    )

    logger.info("STEP 4 | Fitting on training data only")

    train_array = preprocessor.fit_transform(X_train)
    test_array = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        train_array,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        test_array,
        columns=feature_names,
        index=X_test.index
    )

    logger.info(
        "PREPROCESSING DONE | %d features -> %d features",
        len(features),
        len(feature_names)
    )

    return PreparedData(
        X_train_raw=X_train,
        X_test_raw=X_test,
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )


# =====================================================
# 3. EDA UTILITIES
# =====================================================

def records(df: pd.DataFrame, limit=8) -> list:
    return json.loads(
        df.head(limit).to_json(orient="records")
    )


def distribution(y: pd.Series) -> dict:
    return {
        "negative": int((y == 0).sum()),
        "positive": int((y == 1).sum()),
        "positive_rate": round(float(y.mean()) * 100, 2)
    }


def histogram(series: pd.Series) -> list:
    values = pd.to_numeric(
        series, errors="coerce"
    ).dropna().to_numpy(dtype=float)

    if len(values) == 0:
        return []

    counts, edges = np.histogram(values, bins=12)

    return [
        {
            "label": f"{edges[i]:.2f} à {edges[i + 1]:.2f}",
            "count": int(counts[i])
        }
        for i in range(len(counts))
    ]


def numeric_statistics(
    data: PreparedData
) -> list:
    result = []

    for feature in data.numeric_features:
        before = data.X_train_raw[feature].dropna()
        after = data.X_train[f"num__{feature}"]

        result.append({
            "feature": feature,
            "mean_before": round(float(before.mean()), 3),
            "std_before": round(float(before.std(ddof=0)), 3),
            "mean_after": round(float(after.mean()), 3),
            "std_after": round(float(after.std(ddof=0)), 3)
        })

    return result


def categorical_statistics(
    data: PreparedData
) -> tuple[list, dict]:
    if not data.categorical_features:
        return [], {}

    encoder = (
        data.preprocessor
        .named_transformers_["cat"]
        .named_steps["encoder"]
    )

    names = encoder.get_feature_names_out(
        data.categorical_features
    )

    summary = []
    encoded_columns = {}

    offset = 0

    for feature, categories in zip(
        data.categorical_features,
        encoder.categories_
    ):
        count = len(categories)

        columns = [
            f"cat__{name}"
            for name in names[offset:offset + count]
        ]

        encoded_columns[feature] = columns

        summary.append({
            "feature": feature,
            "categories": count,
            "generated_columns": len(columns)
        })

        offset += count

    return summary, encoded_columns


def categorical_inspection(
    data: PreparedData,
    feature: str,
    encoded_columns: dict
) -> dict | None:

    if feature not in data.categorical_features:
        return None

    before_counts = (
        data.X_train_raw[feature]
        .fillna("Manquant")
        .astype(str)
        .value_counts()
    )

    before = [
        {
            "label": str(label),
            "count": int(count)
        }
        for label, count in before_counts.items()
    ]

    columns = encoded_columns[feature]

    after = [
        {
            "label": col.removeprefix(f"cat__{feature}_"),
            "count": int(data.X_train[col].sum())
        }
        for col in columns
    ]

    return {
        "feature": feature,
        "before": before,
        "after": after
    }


# =====================================================
# 4. COMPLETE PREPROCESSING REPORT
# =====================================================

def build_preprocessing_report(
    context: DatasetContext,
    config: PreprocessingConfig
) -> dict:

    data = prepare_dataset(context, config)

    missing_by_feature = [
        {
            "feature": feature,
            "before": int(
                data.X_train_raw[feature].isna().sum()
            ),
            "after": 0
        }
        for feature in data.X_train_raw.columns
    ]

    encoded_summary, encoded_columns = (
        categorical_statistics(data)
    )

    num_feature = config.inspect_numeric

    if num_feature not in data.numeric_features:
        num_feature = (
            data.numeric_features[0]
            if data.numeric_features else None
        )

    cat_feature = config.inspect_categorical

    if cat_feature not in data.categorical_features:
        cat_feature = (
            data.categorical_features[0]
            if data.categorical_features else None
        )

    numeric_inspection = None

    if num_feature:
        numeric_inspection = {
            "feature": num_feature,
            "before": histogram(
                data.X_train_raw[num_feature]
            ),
            "after": histogram(
                data.X_train[f"num__{num_feature}"]
            )
        }

    logger.info("STEP 5 | Generating preprocessing EDA report")

    return {
        "dataset": {
            "dataset_id": context.dataset_id,
            "target": context.target,
            "positive_class": context.positive_class,
            "negative_class": context.negative_class
        },
        "dimensions": {
            "original_rows": len(context.df),
            "train_rows": len(data.X_train),
            "test_rows": len(data.X_test),
            "input_features": len(data.X_train_raw.columns),
            "output_features": len(data.X_train.columns)
        },
        "split": {
            "train": distribution(data.y_train),
            "test": distribution(data.y_test)
        },
        "features": {
            "numeric": data.numeric_features,
            "categorical": data.categorical_features,
            "final": data.X_train.columns.tolist()
        },
        "missing": {
            "before_count": int(
                data.X_train_raw.isna().sum().sum()
            ),
            "after_count": int(
                data.X_train.isna().sum().sum()
            ),
            "by_feature": missing_by_feature
        },
        "encoding": {
            "summary": encoded_summary,
            "inspection": categorical_inspection(
                data,
                cat_feature,
                encoded_columns
            )
        },
        "scaling": {
            "enabled": config.scale_numeric,
            "statistics": numeric_statistics(data),
            "inspection": numeric_inspection
        },
        "preview": {
            "before": records(data.X_train_raw),
            "after": records(data.X_train)
        }
    }
