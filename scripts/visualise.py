#!/usr/bin/env python3

import pyspark
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("E-commerce Data Preprocessing") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

auto_df = spark.read.format("csv").option("header", True).load("hdfs://namenode:9000/project-data/preprocessed.csv")
auto_df.show()

pdf = auto_df.sample(fraction=0.01, seed=42).toPandas()
pdf.price = pdf.price.apply(float)

spark.stop()


pdf.price.plot.hist(column='price')
plt.savefig("figures/price-hist.png")

pdf.category.value_counts().plot.bar()
plt.savefig("figures/cat-bar.png")

pdf.plot.box(by='category', column='price', rot=90)
plt.savefig("figures/cat-price.png")

pdf.brand.value_counts().head().plot.bar(rot=90)
plt.savefig("figures/brand-bar.png")

pdf.product_name.value_counts().head().plot.bar(rot=90)
plt.savefig("figures/prod-bar.png")
