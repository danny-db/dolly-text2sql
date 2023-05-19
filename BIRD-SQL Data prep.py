# Databricks notebook source
# MAGIC %pip install -U datasets

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dolly_df = spark.read.format("json").load("/FileStore/danny_wong/databricks_dolly_15k.jsonl")

display(dolly_df)

# COMMAND ----------

dolly_df.count()

# COMMAND ----------

bird_df = spark.read.option("multiline","true").format("json").load("/FileStore/danny_wong/train.json")

display(bird_df)

# COMMAND ----------

bird_df.count()

# COMMAND ----------

bird_clean_df = (
  bird_df.withColumnRenamed("SQL","response")
  .withColumnRenamed("db_id", "category")
  .withColumnRenamed("question", "instruction")
  .withColumnRenamed("evidence", "context")
  .drop("SQL_toks")
  .drop("question_toks")
  .drop("evidence_toks")
)

display(bird_clean_df)

# COMMAND ----------

spider_df = spark.read.option("multiline","true").format("json").load("/FileStore/danny_wong/train_spider.json")

display(spider_df)

# COMMAND ----------

from pyspark.sql.types import StringType
from pyspark.sql.functions import col,lit

spider_clean_df = (
  spider_df.withColumnRenamed("query","response")
  .withColumnRenamed("db_id", "category")
  .withColumnRenamed("question", "instruction")
  .withColumn("context", lit(None).cast(StringType()))
  .drop("query_toks")
  .drop("question_toks")
  .drop("query_toks_no_value")
  .drop("sql")
)

display(spider_clean_df)

# COMMAND ----------

bird_n_spider = bird_clean_df.unionByName(spider_clean_df)

display(bird_n_spider)

# COMMAND ----------

bird_n_spider.count()

# COMMAND ----------

from datasets import Dataset

bird_n_spider_ds = Dataset.from_spark(bird_n_spider)

# COMMAND ----------


