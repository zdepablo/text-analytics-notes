// Databricks notebook source exported at Tue, 29 Mar 2016 22:51:08 UTC
val training = sqlContext.read.parquet("/databricks-datasets/news20.binary/data-001/training").cache()
val test = sqlContext.read.parquet("/databricks-datasets/news20.binary/data-001/test").cache()

// COMMAND ----------

import org.apache.spark.ml.feature._
import org.apache.spark.ml.clustering.KMeans


val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("words").setPattern("\\s+")
val stop = new StopWordsRemover().setInputCol("words").setOutputCol("filteredWords")
val tf = new CountVectorizer().setInputCol("filteredWords").setOutputCol("rawFeatures").setVocabSize(10000).setMinDF(2)
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("idfFeatures")
val normalizer = new Normalizer().setInputCol("idfFeatures").setOutputCol("features")
val kmeans = new KMeans().setK(20).setMaxIter(100)


// COMMAND ----------

kmeans.explainParams()

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineStage


val stages: Array[PipelineStage] = Array(tokenizer, stop, tf, idf, normalizer, kmeans)
val pipeline = new Pipeline().setStages(stages)

// COMMAND ----------

val model = pipeline.fit(training)

// COMMAND ----------

transformedTest.printSchema()

// COMMAND ----------

val transformedTrain = model.transform(training)

// COMMAND ----------

display(transformedTrain.groupBy("prediction").count())

// COMMAND ----------

import org.apache.spark.ml.clustering.KMeansModel
val stage = model.stages(4)

stage.asInstanceOf[KMeansModel].clusterCenters.foreach(println)
