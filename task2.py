'''
DSCI 553 | Foundations and Applications of Data Mining
Homework 4
Matheus Schmitz
'''

import sys
import os
import time
import random
from pyspark import SparkContext, SparkConf
from collections import defaultdict
from operator import add
from itertools import combinations


def GN(root_node):
	# Track the depth level on the tree
	depth = 0

	# Use dictionaries to construct a tree connecting nodes
	tree, parent_to_child, child_to_parent, distances, node_scores, explored_nodes = {}, {}, defaultdict(set), {}, {}, set()

	# Draw the beginning of the tree
	while depth == 0:

		# Update the tree with the nodes at this level
		tree[depth] = root_node

		# Update the list of used nodes to include the ones from this iteration
		explored_nodes.add(root_node)

		# Find the nodes at the next depth
		next_depth_nodes = adjacency_dict[root_node] #adjacency_dict.value[root_node]

		# Update the dict mapping the current node to its children (the yet unexplored nodes)
		parent_to_child[root_node] = next_depth_nodes

		# Update the dict mapping the current children to its parent node
		[child_to_parent[child].add(parent) for parent, all_childs in parent_to_child.items() for child in all_childs]

		# The distance from the root to all its direct children is 1
		distances[root_node] = 1
	
		# Move to the next depth
		depth += 1


	# Iteratively explore the tree's depth until all nodes have been explored
	while len(next_depth_nodes) > 0:

		# Update the tree with the nodes at this level
		tree[depth] = next_depth_nodes

		# Update the list of used nodes to include the ones from the new iteration
		explored_nodes.update(next_depth_nodes)

		# For all current nodes, get their children
		iteration_nodes = set()
		for node in next_depth_nodes:

			# Find all nodes that share edges with the current node
			connected_nodes = adjacency_dict[node]

			# From those, keep the ones that have not yet been explored
			child_nodes = connected_nodes - explored_nodes

			# Update the dict mapping the current node to its children (the yet unexplored nodes)
			parent_to_child[node] = child_nodes

			# Update the dict mapping the current children to its parent node
			[child_to_parent[child].add(parent) for parent, all_childs in parent_to_child.items() for child in all_childs]

			# Get all parents of the current node
			parent_nodes = child_to_parent[node]

			# The denominator is the sum of the shortest distance of each parent to the root
			if len(parent_nodes) > 0:
				distances[node] = sum([distances[parent] for parent in parent_nodes])
			else:
				distances[node] = 1

			# Go adding all newly discovered nodes to the list of nodes to be ran in the next iterations
			iteration_nodes.update(connected_nodes)

		# Get the nodes to be evaluated on the next depth level
		next_depth_nodes = iteration_nodes - explored_nodes
		
		# Move to the next depth
		depth += 1


	# Give all nodes (except the root) their starting score of 1.0
	for node in nodes:
		if node == root_node:
			node_scores[node] = 0.0
		else:
			node_scores[node] = 1.0

	# Calculate edge betweenness from the bottom up		
	while depth > 1:

		# For each node in the current depth (starting from the deepest point)
		for node in tree[depth-1]:
			
			# Iterate over the node's parents
			for parent in child_to_parent[node]:
				
				# Calculate their betweennes
				child_parent_betweennees = node_scores[node] * (distances[parent] / distances[node]) 
				
				# Update the parent's score (which will matter when comparing it to its parents, aka the grandparents)
				node_scores[parent] += child_parent_betweennees
				
				# Output the betweenness of the two current nodes (child and parent)
				yield (tuple(sorted((node, parent))), child_parent_betweennees)
		
		# Iteratively go upper in depth until the root is reached		
		depth -= 1

	#return [(node_pair, betweenness) for node_pair, betweenness in edge_strength.items()]


def compute_communities(adjacency_dict, nodes):
	# Objects to store data over iterations
	communities = []
	remaning_nodes = nodes.copy()

	# Generate nodes until there are no more "remaining_nodes"
	while len(remaning_nodes) > 0:

		# Pick one node at random as seed and generate communities
		curr_community = compute_community(random.sample(remaning_nodes, 1)[0], adjacency_dict)
		
		# Then update the objects storing the status of the nodes and communities
		communities.append(curr_community)
		remaning_nodes.difference_update(curr_community)

	return communities


def compute_community(root_node, adjacency_dict):
	community = set()
	#community = {root_node}

	# Find all nodes that share edges with the current node
	connected_nodes = adjacency_dict[root_node]

	# Recursively explore all connected nodes
	while len(connected_nodes) > 0:
		
		# Update the list of used nodes to include the ones from the new iteration
		community.update(connected_nodes)

		# For all current nodes, get their children
		iteration_nodes = set()
		for node in connected_nodes:

			# Get the next set of nodes to explore
			connected_nodes_next = adjacency_dict[node]
			iteration_nodes.update(connected_nodes_next)

		# Find the next set of yet unseen connected nodes
		connected_nodes = iteration_nodes - community

	# Handle unconnected nodes
	if len(community) == 0:
		community = {root_node}

	return community


if __name__ == "__main__":

	start_time = time.time()

	# Get user inputs
	filter_threshold = int(sys.argv[1])
	input_file_path = sys.argv[2]
	betweenness_output_file_path = sys.argv[3]
	community_output_file_path = sys.argv[4]

	# Initialize Spark with the 4 GB memory parameters from HW1
	sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))
	sc.setLogLevel("WARN")

	# Read the CSV skipping its header, and reshape it as (user_a, user_b)
	inputRDD = sc.textFile(input_file_path)
	inputHeader = inputRDD.first()
	inputRDD = inputRDD.filter(lambda row: row != inputHeader).map(lambda row: row.split(',')).groupByKey().map(lambda x: (x[0], set(x[1]))).collect()

	# Convert users to nodes and create edges between those whose shared business cross the threshold
	nodes = set()
	edges = set()
	for user_a in inputRDD:
		for user_b in inputRDD:
			# Ensure we don't compare an user against itself
			if user_a[0] != user_b[0]:
				# Then check if the intersection of rated businesses is above the threshold
				if len(user_a[1].intersection(user_b[1])) >= filter_threshold:
					# If so, add both users to the set of nodes
					nodes.update([user_a[0], user_b[0]])
					# And also add an edge between the users
					edges.add(tuple((user_a[0], user_b[0])))
					edges.add(tuple((user_b[0], user_a[0])))

	# For each node, get a set of all other nodes it connects to (it has edges with)
	adjacency_dict = {}
	for (left_node, right_node) in edges:
		if adjacency_dict.get(left_node) == None:
			adjacency_dict[left_node] = set()
		adjacency_dict[left_node].add(right_node)
	#adjacency_dict = sc.broadcast(adjacency_dict)

	# Apply the Girvan Newman algorithm and find communities
	communities = sc.parallelize(nodes).map(lambda n: GN(n))

	# From the communities, calculate betweennees of nodes (dividing by two to accout for the two-way road)
	B_score = communities.flatMap(lambda child_parent_betweennees: [*child_parent_betweennees]).reduceByKey(add).map(lambda row: (row[0], row[1]/2)).sortBy(lambda row: (-row[1], row[0])).collect()

	# Save the pairs and thei betweenness to a txt
	with open(betweenness_output_file_path, 'w') as f_out:
		for node_pair, betweenness in B_score:
			f_out.write(str(node_pair) + ',' + str(round(betweenness, 5)) + '\n')


	# Calculate the degree matrix
	DM = {left_node: len(right_node) for left_node, right_node in adjacency_dict.items()}
	
	# Calculate the adjacency matrix
	AM = {(left_node, right_node): 1 if (left_node, right_node) in edges else 0 for left_node in nodes for right_node in nodes}

	# Calculate m (number of edges in the original graph)
	m = len(edges) / 2

	# Track the number of edges remaning
	remaning_edges = m 

	# Track the modularity
	Q = float('-inf')

	# Iteratively remove edges and track the changes in modularity
	while remaning_edges > 0:

		# Apply the Girvan Newman algorithm and find communities in the ever-shrinking adjacency dict
		new_graph = sc.parallelize(nodes).map(lambda n: GN(n))

		# Recompute betweenness to be used in this iteration
		B_score = new_graph.flatMap(lambda child_parent_betweennees: [*child_parent_betweennees]).reduceByKey(add).map(lambda row: (row[0], row[1]/2)).sortBy(lambda row: (-row[1], row[0])).collect()

		# Take highest "yet unremoved" betweenness
		curr_betweenness = B_score[0][1]

		# Then remove all pairs whose betweenness matches "curr_betweenness"
		for (left_node, right_node), betweenness in B_score:
			if betweenness >= curr_betweenness:
				adjacency_dict[left_node].remove(right_node)
				adjacency_dict[right_node].remove(left_node)
				remaning_edges -= 1

		# Recompute communities taking into consideration the removed links
		iter_communities = compute_communities(adjacency_dict, nodes)

		# Recompute the modularity score Q
		iter_Q = sum([AM[(left_node, right_node)] - DM[left_node] * DM[right_node] / (2 * m) for comm in iter_communities for left_node in comm for right_node in comm]) / (2  * m)

		# If the modularity score improved, store the current Q and communities as the best thus far
		if iter_Q > Q:
			communities = iter_communities
			Q = iter_Q

	# Once all edged have been explores, take the sort the best communities found (necessary for proper outputting)
	output_communities = sc.parallelize(communities).map(lambda community: sorted(community)).sortBy(lambda community: (len(community), community)).collect()
	
	# Save the communities to a txt
	with open(community_output_file_path, "w") as f_out:
		for community in output_communities:
			f_out.write(str(community).replace("[", "").replace("]", "") + "\n")

 	# Close spark context
	sc.stop()

	# Measure the total time taken and report it
	time_elapsed = time.time() - start_time
	print(f'Duration: {time_elapsed}')