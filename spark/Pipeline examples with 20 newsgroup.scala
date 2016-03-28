// Databricks notebook source exported at Mon, 28 Mar 2016 21:49:47 UTC
// MAGIC %fs ls /databricks-datasets/news20.binary/data-001

// COMMAND ----------

val training = sqlContext.read.parquet("/databricks-datasets/news20.binary/data-001/training").cache()

// COMMAND ----------

display(training.limit(10))

// COMMAND ----------

display(training.groupBy("topic").count())

// COMMAND ----------

display(training.groupBy("label").count())

// COMMAND ----------

val test = sqlContext.read.parquet("/databricks-datasets/news20.binary/data-001/test").cache()

// COMMAND ----------

import org.apache.spark.ml.feature._

val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("words").setPattern("\\s+")

// COMMAND ----------

val wordsData = tokenizer.transform(training)

// COMMAND ----------

display(wordsData.limit(10))

// COMMAND ----------

val tfCounter = new CountVectorizer()
  .setInputCol("words")
  .setOutputCol("rawFeatures")
//  .setVocabSize(3)
   .setVocabSize(10000)
  .setMinDF(2)

// COMMAND ----------

val tfCounterModel = tfCounter.fit(wordsData)

// COMMAND ----------

val featuresData = tfCounterModel.transform(wordsData)

// COMMAND ----------

display(featuresData.select("rawFeatures").limit(10))

// COMMAND ----------

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featuresData)
val rescaledData = idfModel.transform(featuresData)


// COMMAND ----------

display(rescaledData.select("label","features").take(10))

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineStage


val prueba: Array[PipelineStage] = Array(tokenizer, tfCounter, idf)

// COMMAND ----------


val pipeline = new Pipeline().setStages(prueba)

// COMMAND ----------

val pipelineModel = pipeline.fit(training)

// COMMAND ----------

val transformedData = pipelineModel.transform(training)

// COMMAND ----------

display(transformedData.take(10))

// COMMAND ----------

transformedData.printSchema()

// COMMAND ----------


