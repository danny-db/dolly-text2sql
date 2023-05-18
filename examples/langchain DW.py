# Databricks notebook source
# MAGIC %md
# MAGIC ## Langchain Example
# MAGIC
# MAGIC This takes a pretrained Dolly model, either from Hugging face or from a local path, and uses langchain
# MAGIC to run generation.
# MAGIC
# MAGIC The model to load for generation is controlled by `input_model`.  The default options are the pretrained
# MAGIC Dolly models shared on Hugging Face.  Alternatively, the path to a local model that has been trained using the
# MAGIC `train_dolly` notebook can also be used.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

default_model = "databricks/dolly-v2-3b"

suggested_models = [
    "databricks/dolly-v1-6b",
    "databricks/dolly-v2-3b",
    "databricks/dolly-v2-7b",
    "databricks/dolly-v2-12b",
]

dbutils.widgets.combobox("input_model", default_model, suggested_models, "input_model")

# COMMAND ----------

from training.generate import InstructionTextGenerationPipeline, load_model_tokenizer_for_generate

input_model = dbutils.widgets.get("input_model")

#model, tokenizer = load_model_tokenizer_for_generate(input_model)
model, tokenizer = load_model_tokenizer_for_generate("/dbfs/dolly_training/dolly__2023-05-18T07:35:00")

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(
    pipeline=InstructionTextGenerationPipeline(
        # Return the full text, because this is what the HuggingFacePipeline expects.
        model=model, tokenizer=tokenizer, return_full_text=True, task="text-generation"))

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

# COMMAND ----------

# Examples from https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html
instructions = [
    "Explain to me the difference between nuclear fission and fusion.",
    "Give me a list of 5 science fiction books I should read next.",
]

# Use the model to generate responses for each of the instructions above.
for instruction in instructions:
    response = llm_chain.predict(instruction=instruction)
    print(f"Instruction: {instruction}\n\n{response}\n\n-----------\n")

# COMMAND ----------

context = (
    """George Washington (February 22, 1732[b] – December 14, 1799) was an American military officer, statesman, """
    """and Founding Father who served as the first president of the United States from 1789 to 1797. Appointed by """
    """the Continental Congress as commander of the Continental Army, Washington led Patriot forces to victory in """
    """the American Revolutionary War and served as president of the Constitutional Convention of 1787, which """
    """created and ratified the Constitution of the United States and the American federal government. Washington """
    """has been called the "Father of his Country" for his manifold leadership in the nation's founding."""
)

instruction = "When did George Washinton serve as president of the Constitutional Convention?"

response = llm_context_chain.predict(instruction=instruction, context=context)
print(f"Instruction: {instruction}\n\nContext:\n{context}\n\nResponse:\n{response}\n\n-----------\n")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Text to SQL test

# COMMAND ----------

# MAGIC %pip install -U sqlalchemy sqlalchemy-databricks

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from training.generate import InstructionTextGenerationPipeline, load_model_tokenizer_for_generate

input_model = dbutils.widgets.get("input_model")

#model, tokenizer = load_model_tokenizer_for_generate(input_model)
model, tokenizer = load_model_tokenizer_for_generate("/dbfs/dolly_training/dolly__2023-05-18T07:35:00")

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(
    pipeline=InstructionTextGenerationPipeline(
        # Return the full text, because this is what the HuggingFacePipeline expects.
        model=model, tokenizer=tokenizer, return_full_text=True, task="text-generation"))

# llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
# llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

# COMMAND ----------

from sqlalchemy.engine import create_engine
from langchain import SQLDatabase, SQLDatabaseChain

table = "gnaf_vic_gold"
schema = "dannywong"

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
# Obtain this from a SQL endpoint under "Connection Details", HTTP Path
endpoint_http_path = "/sql/1.0/warehouses/4c770452d7b840f5"

engine = create_engine(
  f"databricks+connector://token:{databricks_token}@{workspace_url}:443/{schema}",
  connect_args={"http_path": endpoint_http_path}
)

db = SQLDatabase(engine, schema=None, include_tables=[table]) # schema=None to work around https://github.com/hwchase17/langchain/issues/2951 ?
dolly_chain = SQLDatabaseChain(llm=hf_pipeline, database=db, verbose=True)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dannywong.gnaf_vic_gold

# COMMAND ----------

dolly_chain.run("What is the number of records?")

# COMMAND ----------

dolly_chain.run("How many records with the postcode 3128?")

# COMMAND ----------

dolly_chain.run("How many address label with the keyword 'GRANDVIEW RD'?  Can you list them all out?")

# COMMAND ----------


