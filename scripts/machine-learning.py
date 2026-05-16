#!/usr/bin/env python3

import argparse
import os
import sys
import logging
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)

from pyspark.ml import Pipeline

from pyspark.ml.regression import LinearRegression

from pyspark.ml.classification import LogisticRegression

from pyspark.ml.evaluation import (
    RegressionEvaluator,
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

from pyspark.ml.fpm import FPGrowth


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# Utility Functions
# =========================================================

def create_output_dirs():
    dirs = [
        "outputs",
        "outputs/plots",
        "outputs/csv",
        "outputs/models"
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)


def validate_columns(df, required_columns):
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def evaluate_classifier(
    preds,
    name,
    bin_evaluator,
    mc_evaluator_acc,
    mc_evaluator_f1,
    mc_evaluator_prec,
    mc_evaluator_rec
):
    auc = bin_evaluator.evaluate(preds)
    acc = mc_evaluator_acc.evaluate(preds)
    f1 = mc_evaluator_f1.evaluate(preds)
    prec = mc_evaluator_prec.evaluate(preds)
    rec = mc_evaluator_rec.evaluate(preds)

    print(f"\n=== {name} — Test Set Results ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC-ROC": auc
    }


def confusion_matrix_spark(preds, model_name):
    cm = (
        preds
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .toPandas()
    )

    print(f"\n=== Confusion Matrix — {model_name} ===")
    print(cm.to_string(index=False))

    return cm


def cm_to_matrix(cm_pd):
    mat = np.zeros((2, 2), dtype=int)

    for _, row in cm_pd.iterrows():
        mat[int(row["label"])][int(row["prediction"])] += int(row["count"])

    return mat


# =========================================================
# Main Pipeline
# =========================================================

def main(data_path):

    create_output_dirs()

    logger.info("Initializing Spark Session...")

    spark = (
        SparkSession.builder
        .appName("EcommerceSparkMLPipeline")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:

        # =====================================================
        # Load Data
        # =====================================================

        logger.info(f"Loading dataset: {data_path}")

        df = spark.read.csv(
            data_path,
            header=True,
            inferSchema=True
        )

        logger.info(f"Rows: {df.count():,}")
        logger.info(f"Columns: {len(df.columns)}")

        required_columns = [
            "event_type",
            "price",
            "brand",
            "category",
            "hour",
            "day",
            "month",
            "user_session"
        ]

        validate_columns(df, required_columns)

        # =====================================================
        # Feature Engineering
        # =====================================================

        logger.info("Preparing features...")

        data = (
            df.select(
                "event_type",
                "price",
                "brand",
                "category",
                "hour",
                "day",
                "month"
            )
            .dropna()
        )

        brand_indexer = StringIndexer(
            inputCol="brand",
            outputCol="brand_idx",
            handleInvalid="keep"
        )

        category_indexer = StringIndexer(
            inputCol="category",
            outputCol="category_idx",
            handleInvalid="keep"
        )

        event_indexer = StringIndexer(
            inputCol="event_type",
            outputCol="label",
            handleInvalid="keep"
        )

        encoder = OneHotEncoder(
            inputCols=["brand_idx", "category_idx"],
            outputCols=["brand_vec", "category_vec"]
        )

        assembler = VectorAssembler(
            inputCols=[
                "brand_vec",
                "category_vec",
                "hour",
                "day",
                "month"
            ],
            outputCol="features_raw"
        )

        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=False
        )

        preprocessing = Pipeline(
            stages=[
                brand_indexer,
                category_indexer,
                event_indexer,
                encoder,
                assembler,
                scaler
            ]
        )

        preprocessor = preprocessing.fit(data)

        prepared = preprocessor.transform(data)

        # =====================================================
        # REGRESSION
        # =====================================================

        logger.info("Starting regression task...")

        reg_data = (
            prepared
            .select("features", "price")
            .withColumnRenamed("price", "label")
        )

        reg_train, reg_test = reg_data.randomSplit(
            [0.8, 0.2],
            seed=42
        )

        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=20,
            regParam=0.1,
            elasticNetParam=0
        )

        lr_model = lr.fit(reg_train)

        lr_preds = lr_model.transform(reg_test)

        rmse_eval = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="rmse"
        )

        mae_eval = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="mae"
        )

        r2_eval = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="r2"
        )

        lr_rmse = rmse_eval.evaluate(lr_preds)
        lr_mae = mae_eval.evaluate(lr_preds)
        lr_r2 = r2_eval.evaluate(lr_preds)

        logger.info(
            f"Regression Results | "
            f"RMSE={lr_rmse:.4f} | "
            f"MAE={lr_mae:.4f} | "
            f"R2={lr_r2:.4f}"
        )

        # =====================================================
        # CLASSIFICATION
        # =====================================================

        logger.info("Starting classification task...")

        assembler_clf = VectorAssembler(
            inputCols=[
                "brand_vec",
                "category_vec",
                "price",
                "hour",
                "day",
                "month"
            ],
            outputCol="features_raw_clf"
        )

        scaler_clf = StandardScaler(
            inputCol="features_raw_clf",
            outputCol="features_clf",
            withStd=True,
            withMean=False
        )

        binary_clf_data = prepared.filter(
            F.col("event_type").isin("purchase", "cart")
        )

        binary_label_indexer = StringIndexer(
            inputCol="event_type",
            outputCol="binary_label",
            handleInvalid="error"
        )

        binary_clf_data = (
            binary_label_indexer
            .fit(binary_clf_data)
            .transform(binary_clf_data)
        )

        clf_prep = Pipeline(
            stages=[
                assembler_clf,
                scaler_clf
            ]
        ).fit(binary_clf_data)

        clf_data = (
            clf_prep
            .transform(binary_clf_data)
            .select("features_clf", "binary_label")
            .withColumnRenamed("features_clf", "features")
            .withColumnRenamed("binary_label", "label")
        )

        clf_train, clf_test = clf_data.randomSplit(
            [0.8, 0.2],
            seed=42
        )

        log_reg = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=20,
            regParam=0.1
        )

        log_reg_model = log_reg.fit(clf_train)

        log_reg_preds = log_reg_model.transform(clf_test)

        bin_eval = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        acc_eval = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )

        f1_eval = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )

        prec_eval = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )

        rec_eval = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall"
        )

        clf_metrics = evaluate_classifier(
            log_reg_preds,
            "Logistic Regression",
            bin_eval,
            acc_eval,
            f1_eval,
            prec_eval,
            rec_eval
        )

        lr_cm = confusion_matrix_spark(
            log_reg_preds,
            "Logistic Regression"
        )

        # =====================================================
        # Save Models
        # =====================================================

        logger.info("Saving trained models...")

        lr_model.write().overwrite().save(
            "outputs/models/linear_regression_model"
        )

        log_reg_model.write().overwrite().save(
            "outputs/models/logistic_regression_model"
        )

        # =====================================================
        # Visualization
        # =====================================================

        logger.info("Generating plots...")

        preds_pd = (
            lr_preds
            .select("label", "prediction")
            .sample(fraction=0.3, seed=42)
            .toPandas()
        )

        if not preds_pd.empty:
            # Actual vs Predicted
            plt.figure(figsize=(8, 6))

            plt.scatter(
                preds_pd["label"],
                preds_pd["prediction"],
                alpha=0.4,
                s=20
            )

            max_val = max(
                preds_pd["label"].max(),
                preds_pd["prediction"].max()
            )

            plt.plot(
                [0, max_val],
                [0, max_val],
                "r--"
            )

            plt.xlabel("Actual Price")
            plt.ylabel("Predicted Price")

            plt.title(
                f"Linear Regression (R²={lr_r2:.3f})"
            )

            plt.tight_layout()

            plt.savefig(
                "outputs/plots/reg_actual_vs_predicted.png",
                dpi=150
            )

            plt.close()

            # Residuals
            preds_pd["residual"] = (
                preds_pd["label"] -
                preds_pd["prediction"]
            )

            plt.figure(figsize=(8, 5))

            plt.hist(
                preds_pd["residual"],
                bins=50
            )

            plt.axvline(0, linestyle="--")

            plt.xlabel("Residual")
            plt.ylabel("Frequency")

            plt.title("Residual Distribution")

            plt.tight_layout()

            plt.savefig(
                "outputs/plots/reg_residuals.png",
                dpi=150
            )

            plt.close()
        else:
            logger.warning("Sampled regression predictions DataFrame is empty; skipping regression plots.")

        # Confusion Matrix
        mat = cm_to_matrix(lr_cm)

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            mat,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(
            "outputs/plots/confusion_matrix.png",
            dpi=150
        )

        plt.close()

        # =====================================================
        # Save Metrics
        # =====================================================

        pd.DataFrame([
            {
                "Model": "Linear Regression",
                "RMSE": lr_rmse,
                "MAE": lr_mae,
                "R2": lr_r2
            }
        ]).to_csv(
            "outputs/csv/regression_results.csv",
            index=False
        )

        pd.DataFrame([clf_metrics]).to_csv(
            "outputs/csv/classification_results.csv",
            index=False
        )

        # =====================================================
        # FP-Growth
        # =====================================================

        logger.info("Starting FP-Growth...")

        item_column = "category"

        events = (
            df.filter(F.col("event_type") == "purchase")
            .dropna(subset=["user_session", item_column])
        )

        baskets = (
            events.groupBy("user_session")
            .agg(
                F.collect_set(item_column).alias("items")
            )
            .filter(F.size("items") >= 2)
        )

        basket_count = baskets.count()

        if basket_count > 0:

            fpgrowth = FPGrowth(
                itemsCol="items",
                minSupport=0.01,
                minConfidence=0.30
            )

            fp_model = fpgrowth.fit(baskets)

            freq_itemsets = (
                fp_model.freqItemsets
                .withColumn(
                    "itemset_size",
                    F.size("items")
                )
                .withColumn(
                    "support",
                    F.col("freq") / basket_count
                )
            )

            rules = fp_model.associationRules

            freq_itemsets_pd = (
                freq_itemsets
                .withColumn(
                    "items_str",
                    F.concat_ws(" | ", "items")
                )
                .toPandas()
            )

            rules_pd = (
                rules
                .withColumn(
                    "antecedent_str",
                    F.concat_ws(" + ", "antecedent")
                )
                .withColumn(
                    "consequent_str",
                    F.concat_ws(" + ", "consequent")
                )
                .toPandas()
            )

            freq_itemsets_pd.to_csv(
                "outputs/csv/frequent_itemsets.csv",
                index=False
            )

            rules_pd.to_csv(
                "outputs/csv/association_rules.csv",
                index=False
            )

        logger.info("Pipeline completed successfully.")

    except Exception as e:

        logger.exception("Pipeline failed.")
        print(f"\nERROR: {e}")

        sys.exit(1)

    finally:

        logger.info("Stopping Spark session...")
        spark.stop()


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Spark MLlib E-Commerce Pipeline"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV dataset"
    )

    args = parser.parse_args()

    main(args.data)
