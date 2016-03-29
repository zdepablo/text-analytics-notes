// Databricks notebook source exported at Tue, 29 Mar 2016 22:52:12 UTC
val training = sqlContext.read.parquet("/databricks-datasets/news20.binary/data-001/training").cache()
val test = sqlContext.read.parquet("/databricks-datasets/news20.binary/data-001/test").cache()

// COMMAND ----------

import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression

val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("words").setPattern("\\s+")
val tf = new CountVectorizer().setInputCol("words").setOutputCol("rawFeatures").setVocabSize(10000).setMinDF(2)
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01)


// COMMAND ----------

lr.explainParams()

// COMMAND ----------

tf.explainParams()

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineStage


val stages: Array[PipelineStage] = Array(tokenizer, tf, idf, lr)
val pipeline = new Pipeline().setStages(stages)

// COMMAND ----------

val model = pipeline.fit(training)


// COMMAND ----------

val transformedTest = model.transform(test)

// COMMAND ----------

transformedTest.printSchema()

// COMMAND ----------

display(transformedTest.take(10))

// COMMAND ----------

transformedTest.select("prediction").take(10)

// COMMAND ----------

import org.apache.spark.ml.tuning.ParamGridBuilder

val paramGrid = new ParamGridBuilder()
  .addGrid(tf.vocabSize, Array(100, 1000, 10000))
  .addGrid(lr.regParam, Array(0.1, 0.01))
  .build()

// COMMAND ----------

import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val evaluator = new BinaryClassificationEvaluator

val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(10)


// COMMAND ----------

val cvModel = cv.fit(training)

// COMMAND ----------

cvModel.transform(test)
  .select("label", "prediction")
  .show()

// COMMAND ----------

cvModel.avgMetrics.foreach(println)

// COMMAND ----------

cvModel.bestModel

// COMMAND ----------

evaluator.evaluate(cvModel.transform(test))
