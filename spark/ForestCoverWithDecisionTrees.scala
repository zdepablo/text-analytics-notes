// Databricks notebook source exported at Sun, 27 Mar 2016 22:18:56 UTC
// MAGIC %md # Clasificación de tipos de bosque
// MAGIC 
// MAGIC **Objetivo** : Demostrar el uso de MLlib en un problema de clasificación. 
// MAGIC 
// MAGIC **Dataset** : El dataset Forest Covtyoe registra el tipo de bosque de diferentes parcelas de terreno (30m x 30m ) en el [Parque Nacional Roosevelt](https://en.wikipedia.org/wiki/Roosevelt_National_Forest) en Colorado, USA. El objetivo de este dataset es poder predecir el tipo de bosque a partir de variables cartográficas de forma que se puedan gestionar los recursos naturales y complementar una catalogación "manual" que pueda ser potencialmente costosa. 
// MAGIC 
// MAGIC Cada parcela se describe mediante una serie de variables independientes (características): 
// MAGIC   - elevación
// MAGIC   - pendiente 
// MAGIC   - distancia hasta puntos de agua
// MAGIC   - sombra
// MAGIC   - zona (4) 
// MAGIC   - tipo de terreno (40) 
// MAGIC   
// MAGIC Por último, la variable a predecir es el tipo de bosque que se encuentra en la parcela de los que se catalogan 7 tipos diferentes: 
// MAGIC 
// MAGIC    - 1 Spruce/Fir - (abeto)
// MAGIC    - 2 Lodgepole Pine 
// MAGIC    - 3 Ponderosa Pine
// MAGIC    - 4 Cottonwood/Willow (álamo/sauce)
// MAGIC    - 5 Aspen  (alamo temblón)
// MAGIC    - 6 Douglas-fir (abeto de Douglas)
// MAGIC    - 7 Krummholz
// MAGIC 
// MAGIC https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/

// COMMAND ----------

// MAGIC %md ##¿Por que elegimos este problema? 
// MAGIC 
// MAGIC - Problema relativamente sencillo
// MAGIC - Caractericas principalmente numéricas 
// MAGIC - Características categóricas ya están transformadas - One Hot Encoding
// MAGIC - Árboles de decision pueden trabajar con las características tal y como se representan
// MAGIC - Usamos RDDs 

// COMMAND ----------

// MAGIC %md ## ¿Cómo funciona un árbol de decision?
// MAGIC 
// MAGIC http://www.r2d3.us/visual-intro-to-machine-learning-part-1/

// COMMAND ----------

// MAGIC %md ### 1. Carga de datos

// COMMAND ----------

val rawData = sc.textFile("/FileStore/tables/nz8hk79k1457221209758/covtype.data")

// COMMAND ----------

rawData.first

// COMMAND ----------

// MAGIC %md Visualizamos las características y en este caso la variable a predecir es la última

// COMMAND ----------

// MAGIC %md ### 2. Transformamos los datos en un RDD de LabeledPoint

// COMMAND ----------

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

val data = rawData.map { line =>
  val values = line.split(',').map(_.toDouble)
  val featureVector = Vectors.dense(values.init)
  val label = values.last - 1
  LabeledPoint(label, featureVector)
}

// COMMAND ----------

data.first

// COMMAND ----------

// MAGIC %md ### 3. Dividimos los datos en conjunto de entrenamiento, test y validación.
// MAGIC - Puesto que no se trata de un conjunto de datos masivo podemos cachearlos en memoria - generalmente no lo haríamos. 

// COMMAND ----------

val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))

trainData.cache()
cvData.cache()
testData.cache()

// COMMAND ----------

trainData.count()

// COMMAND ----------

cvData.count()

// COMMAND ----------

testData.count()

// COMMAND ----------

// MAGIC %md ### 4. Entrenamos el modelo usando un Árbol de decisión

// COMMAND ----------

import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._

val model = DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), "gini", 4, 100)

// COMMAND ----------

// MAGIC %md 
// MAGIC 
// MAGIC - Numero de clases: 7 
// MAGIC - Map[Int,Int]() - ignorar de momento, mapeo de las variables categóricas
// MAGIC - Método para elegir la partición de los nodos: gini 
// MAGIC - Máxima profundidad: 4
// MAGIC - Máximo número de bins a usar en las variables continuas: 100 

// COMMAND ----------

// MAGIC %md ### 5. Predicción con el modelo entrenado

// COMMAND ----------

val predictions = cvData.map(x => model.predict(x.features))

// COMMAND ----------

predictions.take(10)

// COMMAND ----------

val predictionsAndLabels = cvData.map(example => (model.predict(example.features), example.label) )

// COMMAND ----------

predictionsAndLabels.take(10).foreach(println)

// COMMAND ----------

// MAGIC %md ### 6. Definimos una función de evaluación

// COMMAND ----------

import org.apache.spark.mllib.evaluation._

def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
  val predictionsAndLabels = data.map(example => (model.predict(example.features), example.label) )
  
  new MulticlassMetrics(predictionsAndLabels)
}

// COMMAND ----------

val metrics = getMetrics(model, cvData)

// COMMAND ----------

println("Confusion matrix:")
println(metrics.confusionMatrix)

// COMMAND ----------

val precision = metrics.precision
val recall = metrics.recall // same as true positive rate
val f1Score = metrics.fMeasure

// COMMAND ----------

println("Summary Statistics")
println(s"Precision = $precision")
println(s"Recall = $recall")
println(s"F1 Score = $f1Score")

// COMMAND ----------

(0 until 7).map(
cat => (cat, metrics.precision(cat), metrics.recall(cat))
).foreach(println)

// COMMAND ----------

// MAGIC %md 7. ¿Es útil? Mejora respecto al modelo trivial

// COMMAND ----------

def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
 
  val countsByCategory = data.map(_.label).countByValue()
  val counts = countsByCategory.toArray.sortBy(_._1).map(_._2) 
  
 counts.map(_.toDouble / counts.sum)
}

// COMMAND ----------

val trainPriorProbabilities = classProbabilities(trainData)

// COMMAND ----------

val cvPriorProbabilities = classProbabilities(cvData)

// COMMAND ----------

trainPriorProbabilities.zip(cvPriorProbabilities).map {
   case (trainProb, cvProb) => trainProb * cvProb
}.sum

// COMMAND ----------

// MAGIC %md 8. Probamos diferentes hyperparámetros

// COMMAND ----------

val evaluations = for (
  impurity <- Array("gini", "entropy");
  depth <- Array(1, 20);
  bins <- Array(10,300)
) yield {
  
   val model = DecisionTree.trainClassifier( trainData, 7, Map[Int,Int](), impurity, depth, bins)
   val predictionsAndLabels = cvData.map(example => (model.predict(example.features), example.label))
   val accuracy = new MulticlassMetrics(predictionsAndLabels).precision
  
  ((impurity, depth, bins), accuracy)
}

// COMMAND ----------

evaluations.sortBy(_._2).reverse.foreach(println)

// COMMAND ----------

// MAGIC %md En este caso, el parámetro de la profundidad de los árboles es decisivo para encontrar un buen clasificador

// COMMAND ----------

val bestModel = DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), "entropy", 20, 300)
val testMetrics = getMetrics(bestModel, cvData)

// COMMAND ----------

val precision = testMetrics.precision
val recall = testMetrics.recall // same as true positive rate
val f1Score = testMetrics.fMeasure
