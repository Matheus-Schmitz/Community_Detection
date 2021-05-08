name := "hw4"

version := "0.1"

scalaVersion := "2.11.12"

val sparkVersion = "2.4.4"

resolvers ++= Seq(
  "apache-snapshots" at "https://repository.apache.org/snapshots/",
  "SparkPackages" at "https://dl.bintray.com/spark-packages/maven/"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion,
  "org.apache.spark" %% "spark-graphx" % sparkVersion,
  "graphframes" % "graphframes" % "0.6.0-spark2.3-s_2.11"
)
