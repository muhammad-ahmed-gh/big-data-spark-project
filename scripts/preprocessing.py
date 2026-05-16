import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import trim, lower, col, sum, when, split, hour, dayofweek, month, year

# === Initialize SparkSession ===
spark = SparkSession.builder \
    .appName("E-commerce Data Preprocessing") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

auto_df = spark.read.format("csv").option("header", True).load("hdfs://namenode:9000/project-data/ecommerce.csv")
auto_df.show()


print("============================")
print("     Exploring Metadata     ")
print("============================")

print(f"\nnum rows: {auto_df.count()}")
print("data schema:")
auto_df.printSchema()

null_counts = auto_df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in auto_df.columns
])
print("nulls count:")
null_counts.show()


print("\n============================")
print("       Data Reduction       ")
print("============================")
print("\n=== removing duplicates ===")
before = auto_df.count()
print(f"before: {before}")
df = auto_df.dropDuplicates()
after = df.count()
print(f"after: {after}")
print("no duplicates" if before == after else f"{before - after} duplicates removed")


print("\n=== removing nulls in brand and category ===")
before = after
print(f"before: {before}")
df = df.dropna()
after = df.count()
print(f"after: {after}")
print("no nulls" if before == after else f"{before - after} records dropped")


print("\n=== checking event_type values ===")
df.groupBy("event_type").count().show()
print("\nwe don't need all the records of event_type view")
print("we can take a sample of it instead")

views = df.filter(col("event_type") == "view")
others = df.filter(col("event_type") != "view")
views_sampled = views.sample(fraction=0.2, seed=42)  # keep 20%
df = others.union(views_sampled)
df.groupBy("event_type").count().show()
print(f"rows after: {df.count()}")


print("\n=== casting price to double ===")
df = df.withColumn("price", col("price").cast("double"))
df.printSchema()

print("\n=== normalize strings ===")
df = df.withColumn("category_code", trim(lower(col("category_code"))))
df.show()

print("\n=== creating columns for product name and category ===")
df = df.withColumn("product_name", split(col("category_code"), "\\.").getItem(1))
df = df.withColumn("category", split(col("category_code"), "\\.").getItem(0))
df = df.drop("category_code")
df = df.drop("category_id")
df.show()


print("\n=== checking values of price ===")
df.describe(["price"]).show()
print("no abnormal outliers")

print("\n=== changing schema of timestamp ===")
df = df.withColumn("event_time", col("event_time").cast("timestamp"))
df = df.withColumn("hour", hour(col("event_time")))
df = df.withColumn("day", dayofweek(col("event_time"))) # 1 --> Sunday
df = df.withColumn("month", month(col("event_time")))
df = df.withColumn("year", year(col("event_time")))
df = df.drop("event_time")
df.show()
print("event_time is now easier to deal with")

print("\n============================")
print("    End of Preprocessing    ")
print("============================")
print("saving to hfds://namenode:9000/project-data/preprocessed-data/")
df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("hdfs://namenode:9000/project-data/preprocessed-data")
print("done.")
spark.stop()
