// DSCI 553 | Foundations and Applications of Data Mining
// Homework 4
// Matheus Schmitz
// USC ID: 5039286453

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SparkSession, Row}
import scala.collection.mutable
import java.io._
import scala.util.Random
import org.graphframes._
import org.apache.spark.sql.types._
import scala.math.Ordering.Implicits._


object task1 {

  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()

    // Get user inputs
    val filter_threshold = args(0).toInt
    val input_file_path = args(1)
    val community_output_file_path = args(2)

    // Initialize Spark with the 4 GB memory parameters from HW1
    val config = new SparkConf().setMaster("local[*]").setAppName("task2").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g") //.set("spark.testing.memory", "471859200")
    val SS = SparkSession.builder.config(config).getOrCreate()
    import SS.implicits._
    val sc = SS.sparkContext
    val sql_context = SS.sqlContext
    sc.setLogLevel("WARN")

    // Read the CSV skipping its header, and reshape it as (user_a, user_b)
    val inputRDDwithHeader = sc.textFile(input_file_path)
    val inputHeader = inputRDDwithHeader.first()
    val inputRDD = inputRDDwithHeader.filter(row => row != inputHeader).map(row => (row.split(',')(0), row.split(',')(1))).groupByKey().mapValues(x => x.toSet).collect()

    // Convert users to nodes and create edges between those whose shared business cross the threshold
    val nodes: mutable.Set[String] = mutable.Set.empty[String]
    val edges: mutable.Set[(String, String)] = mutable.Set.empty[(String, String)]
    for (user_a <- inputRDD) {
      for (user_b <- inputRDD) {
        //Ensure we don't compare an user against itself
        if (user_a._1 != user_b._1) {
          // Then check if the intersection of rated businesses is above the threshold
          if (user_a._2.intersect(user_b._2).size >= filter_threshold) {
            // If so, add both users to the set of nodes
            nodes.add(user_a._1)
            nodes.add(user_b._1)
            // And also add an edge between the users
            edges.add((user_a._1, user_b._1))
            edges.add((user_b._1, user_a._1))
          }
        }
      }
    }
    // Convert the data to a Spark Dataframe so that Spark GraphFrames' LPA can be called on it
    // There are specific requirements for the column names according to: https://graphframes.github.io/graphframes/docs/_site/api/python/graphframes.html#graphframes.GraphFrame
    // Spark-Scala documentation on creating DataFrames: https://spark.apache.org/docs/2.4.4/sql-getting-started.html

    // The schema is encoded in a string
    val schemaString_nodes = "id"
    val schemaString_edges = "src,dst"

    // Generate the schema based on the string of schema
    val fields_nodes = schemaString_nodes.split(",").map(fieldName => StructField(fieldName, StringType, nullable = true))
    val fields_edges = schemaString_edges.split(",").map(fieldName => StructField(fieldName, StringType, nullable = true))
    val schema_nodes = StructType(fields_nodes)
    val schema_edges = StructType(fields_edges)

    // Convert records of the RDD (people) to Rows
    val rowRDD_nodes = sc.parallelize(nodes.toSeq).map(attributes => Row(attributes))
    val rowRDD_edges = sc.parallelize(edges.toSeq).map(attributes => Row(attributes._1, attributes._2))

    // Apply the schema to the RDD
    val nodesDF = sql_context.createDataFrame(rowRDD_nodes, schema_nodes)
    val edgesDF = sql_context.createDataFrame(rowRDD_edges, schema_edges)

    // Build a GraphFrame
    val GF = GraphFrame(nodesDF, edgesDF)

    // Apply Label Propagation with maxIter = 5 as specified in the assingment
    val LPA_df = GF.labelPropagation.maxIter(5).run()

    // For each identified community (created under column "label"), create a set of all users in that community
    val	output = LPA_df.rdd.map(row => (row(1).toString, row(0).toString)).groupByKey()

    // Then sort both the users within a community (lexicographically) and then sort the communities themselves (by size, ascending, breaking ties by name of the first user)
    val output_communities = output.map(comm => comm._2.toList.sorted).sortBy(members => (members.size, members)).collect()

    // Save the communities to a txt
    val pw = new PrintWriter(new File(community_output_file_path))
    for (community <- output_communities)
      for (member <- community) {
        pw.write("'" + member + "'")
        if (community.indexOf(member) != community.length - 1)
          pw.write(", ")
        else
          pw.write("\n")
      }
    pw.close()

    // Close spark context
    sc.stop()

    // Measure the total time taken and report it
    val total_time = System.currentTimeMillis() - start_time
    val time_elapsed = (total_time).toFloat / 1000.toFloat
    println("Duration: " + time_elapsed)
  }
}