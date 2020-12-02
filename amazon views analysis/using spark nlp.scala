import org.apache.spark.sql._

import org.apache.spark.sql.types._

import com.johnsnowlabs.nlp.annotator._

import com.johnsnowlabs.nlp.base._

import com.johnsnowlabs.util.Benchmark

import org.apache.spark.ml.Pipeline

import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object TrainViveknSentiment extends App {

  

val spark: SparkSession = SparkSession

                                   .builder()
 
                                   .appName("test")

                                   .master("local[*]")

                                   .config("spark.driver.memory", "4G")
   
                                   .config("spark.kryoserializer.buffer.max","200M")
  
                                   .config("spark.network.timeout", "3000000000")

                                   .config("spark.executor.heartbeatInterval", "200000")
   
                                   .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
   
                                   .getOrCreate()


  
spark.sparkContext.setLogLevel("WARN")
  
  
 
import spark.implicits._


case class AmzView(label : Int, view : String)


val train_path = "/home/harsh/Desktop/amazon views/1305_800230_compressed_train.ft.txt.bz2/train.ft.txt"

val test_path = "/home/harsh/Desktop/amazon views/1305_800230_compressed_test.ft.txt.bz2/test.ft.txt"


def pre_process(path:String) = { 
val data = spark.sparkContext.textFile(path).map(attributes => 
  AmzView(attributes(9), attributes.substring(11, attributes.length()).replace("\"","").toLowerCase()
   
                                                                      .replaceAll("\n", "")
   
                                                                      .replaceAll("rt\\s+", "")
   
                                                                      .replaceAll("\\s+@\\w+", "")
 
                                                                      .replaceAll("@\\w+", "")
  
                                                                      .replaceAll("\\s+#\\w+", "")
  
                                                                      .replaceAll("#\\w+", "")

                                                                      .replaceAll("(?:https?|http?)://[\\w/%.-]+", "")
 
                                                                      .replaceAll("(?:https?|http?)://[\\w/%.-]+\\s+", "")

                                                                      .replaceAll("(?:https?|http?)//[\\w/%.-]+\\s+", "")
 
                                                                      .replaceAll("(?:https?|http?)//[\\w/%.-]+", "")
 
                                                                      .replaceAll("[^\u0000-\uFFFF]","")
  
                                                                      .replaceAll("(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])","") 
   
                                                                      .trim()
 
  )).toDF()


def mo(d:Int) :Int = {if(d==49){2} else{1}}

val lmo = spark.udf.register("mo", mo _)

val p = data.select( (lmo($"label")).alias("label"), $"view")

def mpo(d:Int) :String = {if(d==2){"negative"} else{"positive"}}

val po = spark.udf.register("mpo", mpo _)

val pl = p.select( (po($"label")).alias("sentiment"), $"view", $"label")

pl

}


val training = pre_process(train_path)
 
val testing = pre_process(test_path)

  
  val document = new DocumentAssembler()
    
                      .setInputCol("view")
 
                      .setOutputCol("document")

 
 val token = new Tokenizer()
    
            .setInputCols("document")

            .setOutputCol("token")


  val normalizer = new Normalizer()
  
                   .setInputCols("token")

                   .setOutputCol("normal")


  val vivekn = new ViveknSentimentApproach()
   
                .setInputCols("document", "normal")
 
                .setOutputCol("result_sentiment")

                .setSentimentCol("sentiment")


  val finisher = new Finisher()
 
               .setInputCols("result_sentiment")
 
               .setOutputCols("final_sentiment")


val pipeline = new Pipeline().setStages(Array(document, token, normalizer, vivekn, finisher))


val sparkPipeline = pipeline.fit(training)


def co(s:String):Double = {if(s.equals("negative")){2} else{1}}

val lco = spark.udf.register("co", co _)


val evaluator = new MulticlassClassificationEvaluator()
  
                      .setLabelCol("label")
 
                     .setPredictionCol("prediction") 

                      .setMetricName("accuracy")


val predict_train = sparkPipeline.transform(training).select($"view", $"label", lco($"final_sentiment"(0)).alias("prediction"))

val predict_test = sparkPipeline.transform(testing).select($"view", $"label", lco($"final_sentiment"(0)).alias("prediction"))


val training_accuracy = evaluator.evaluate(predict_train)

val test_accuracy = evaluator.evaluate(predict_test)

println(training_accuracy)

println(test_accuracy)


spark.close()
}
