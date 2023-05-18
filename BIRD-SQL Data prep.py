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

from datasets import Dataset

bird_ds = Dataset.from_spark(bird_clean_df)

# COMMAND ----------


