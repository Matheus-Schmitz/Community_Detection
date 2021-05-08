'''
DSCI 553 | Foundations and Applications of Data Mining
Homework 4
Matheus Schmitz
'''

import sys
import os
import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from graphframes import GraphFrame

# spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1.py <filter threshold> <input_file_path> <community_output_file_path>

#os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

if __name__ == "__main__":

	start_time = time.time()

	# Get user inputs
	filter_threshold = int(sys.argv[1])
	input_file_path = sys.argv[2]
	community_output_file_path = sys.argv[3]

	# Initialize Spark with the 4 GB memory parameters from HW1
	sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))
	sc.setLogLevel("WARN")
	sql_context = SQLContext(sc)

	# Read the CSV skipping its header, and reshape it as (user (bizz1, bizz2, bizz3, ...))
	inputRDD = sc.textFile(input_file_path)
	inputHeader = inputRDD.first()
	inputRDD = inputRDD.filter(lambda row: row != inputHeader).map(lambda row: row.split(',')).groupByKey().map(lambda row: (row[0], set(row[1]))).collect()

	# Convert users to nodes and create edges between those whose shared business cross the threshold
	nodes = set()
	edges = set()
	for user_a in inputRDD:
		for user_b in inputRDD:
			# Ensure we don't compare an user against itself
			if user_a[0] != user_b[0]:
				# Then check if the interction of rated businesses is above the threshold
				if len(user_a[1].intersection(user_b[1])) >= filter_threshold:
					# If so, add both users to the list of nodes
					nodes.update([(user_a[0], ), (user_b[0], )])
					# And also add an edge between the users
					edges.add(tuple((user_a[0], user_b[0])))
					edges.add(tuple((user_b[0], user_a[0])))

	# Convert the data to a Spark Dataframe so that Spark GraphFrames' LPA can be called on it
	# There are specific requirements for the column names according to: https://graphframes.github.io/graphframes/docs/_site/api/python/graphframes.html#graphframes.GraphFrame
	nodes_df = sql_context.createDataFrame(list(nodes), ['id'])
	edges_df = sql_context.createDataFrame(list(edges), ['src', 'dst'])

	# Build a GraphFrame
	GF = GraphFrame(nodes_df, edges_df)

	# Apply Label Propagation with maxIter = 5 as specified in the assingment
	LPA_df = GF.labelPropagation(maxIter=5)

	# For each identified community (created under column "label"), create a set of all users in that community
	output = LPA_df.rdd.map(lambda x: (x['label'], [x['id']])).groupByKey()

	# Then sort both the users within a community (lexicographically) and then sort the communities themselves (by size, ascending, breaking ties by name of the first user)
	output = output.map(lambda comm: list(sorted(comm[1]))).sortBy(lambda members: (len(members), members)).collect()

	with open(community_output_file_path, "w") as f_out:
		for comm in output:
			f_out.write(str(comm).replace("[", "").replace("]", "") + "\n")

	# Close spark context
	sc.stop()

	# Measure the total time taken and report it
	time_elapsed = time.time() - start_time
	print(f'Duration: {time_elapsed}')