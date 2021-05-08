// DSCI 553 | Foundations and Applications of Data Mining
// Homework 4
// Matheus Schmitz
// USC ID: 5039286453

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import scala.collection.mutable
import java.io._
import scala.util.Random


object task2 {

  def GN(root_node: String,
         adjacency_dict: mutable.Map[String, mutable.Set[String]],
         nodes: mutable.Set[String]): mutable.Map[(String, String), Double] = {
    // Track the depth level on the tree
    var depth = 0

    // Use dictionaries to construct a tree connecting nodes
    var tree = mutable.Map.empty[Int, mutable.Set[String]]
    var parent_to_child = mutable.Map.empty[String, mutable.Set[String]]
    var child_to_parent = mutable.Map.empty[String, mutable.Set[String]]
    var distances = mutable.Map.empty[String, Double]
    var node_scores = mutable.Map.empty[String, Double]
    var explored_nodes = mutable.Set.empty[String]
    var next_depth_nodes = mutable.Set.empty[String]
    var betweenness_scores = mutable.Map.empty[(String, String), Double]


    while (depth == 0){

      // Update the tree with the nodes at this level
      tree += (depth -> mutable.Set(root_node))

      // Update the list of used nodes to include the ones from this iteration
      explored_nodes.add(root_node)

      // Find the nodes at the next depth
      next_depth_nodes = adjacency_dict(root_node)

      // Update the dict mapping the current node to its children (the yet unexplored nodes)
      if (!parent_to_child.contains(root_node)) {
        parent_to_child += (root_node -> next_depth_nodes)
      }
      else {
        parent_to_child(root_node) ++= next_depth_nodes
      }

      // Update the dict mapping the current children to its parent node
      for ((parent, all_childs) <- parent_to_child.iterator) {
        for (child <- all_childs) {
          if (!child_to_parent.contains(child)) {
            child_to_parent += (child -> mutable.Set(parent))
          }
          else {
            child_to_parent(child).add(parent)
          }
        }
      }

      // The distance from the root to all its direct children is 1
      distances(root_node) = 1

      // Move to the next depth
      depth += 1
    }

    // Iteratively explore the tree's depth until all nodes have been explored
    while (next_depth_nodes.nonEmpty) {

      // Update the tree with the nodes at this level
      tree += (depth -> next_depth_nodes)

      // Update the list of used nodes to include the ones from the new iteration
      explored_nodes ++= next_depth_nodes

      // For all current nodes, get their children
      var iteration_nodes = mutable.Set.empty[String]
      for (node <- next_depth_nodes){

        // Find all nodes that share edges with the current node
        var connected_nodes = adjacency_dict(node)

        // From those, keep the ones that have not yet been explored
        var child_nodes = connected_nodes.diff(explored_nodes)

        // Update the dict mapping the current node to its children (the yet unexplored nodes)
        if (!parent_to_child.contains(node)) {
          parent_to_child += (node -> child_nodes)
        }
        else {
          parent_to_child(node) ++= child_nodes
        }

        // Update the dict mapping the current children to its parent node
        for ((parent, all_childs) <- parent_to_child.iterator) {
          for (child <- all_childs) {
            if (!child_to_parent.contains(child)) {
              child_to_parent += (child -> mutable.Set(parent))
            }
            else {
              child_to_parent(child).add(parent)
            }
          }
        }

        // Get all parents of the current node
        var parent_nodes = child_to_parent(node)

        //# The denominator is the sum of the shortest distance of each parent to the root
        if (parent_nodes.nonEmpty) {
          var total_dist = 0.toDouble
          for (parent <- parent_nodes){
            total_dist = total_dist + distances(parent)}
          distances(node) = total_dist}
        else {
          distances(node) = 1}

        // Go adding all newly discovered nodes to the list of nodes to be ran in the next iterations
        iteration_nodes ++= connected_nodes
      }
      // Get the nodes to be evaluated on the next depth level
      next_depth_nodes = iteration_nodes.diff(explored_nodes)

      // Move to the next depth
      depth += 1
    }

    // Give all nodes (except the root) their starting score of 1.0
    for (node <- nodes) {
      if (node == root_node){
        node_scores += (node -> 0.toDouble)
      }
      else {
        node_scores += (node -> 1.toDouble)
      }
    }

   // Calculate edge betweenness from the bottom up
   while (depth > 1){

     // For each node in the current depth (starting from the deepest point)
     for (node <- tree(depth-1)){

       // Iterate over the node's parents
       for (parent <- child_to_parent(node)){

         // Calculate their betweennes
         var child_parent_betweennees = node_scores(node) * (distances(parent) / distances(node))

         // Update the parent's score (which will matter when comparing it to its parents, aka the grandparents)
         node_scores(parent) += child_parent_betweennees

         // Output the betweenness of the two current nodes (child and parent), making sure they are sorted
        if (node < parent) {
         betweenness_scores += ((node, parent) -> child_parent_betweennees)
       }
        else {
          betweenness_scores += ((parent, node) -> child_parent_betweennees)
        }
       }
     }
     // Iteratively go upper in depth until the root is reached
     depth -= 1
   }
    return betweenness_scores
  }


  def compute_communities(adjacency_dict: mutable.Map[String, mutable.Set[String]],
                          nodes: mutable.Set[String]): mutable.ListBuffer[mutable.Set[String]] = {

    // Objects to store data over iterations
    var communities: mutable.ListBuffer[mutable.Set[String]] = mutable.ListBuffer.empty[mutable.Set[String]]
    var remaning_nodes: mutable.Set[String] = nodes.clone()

    // Generate nodes until there are no more "remaining_nodes"
    while (remaning_nodes.nonEmpty) {

      // Pick one node at random as seed and generate communities
      var curr_community = compute_community(Random.shuffle(remaning_nodes).head, adjacency_dict)

      // Then update the objects storing the status of the nodes and communities
      communities.append(curr_community)
      remaning_nodes --= curr_community
    }
    return communities
  }


  def compute_community(root_node: String,
                        adjacency_dict: mutable.Map[String, mutable.Set[String]]): mutable.Set[String] = {
    var community: mutable.SortedSet[String] = mutable.SortedSet.empty[String]
    community += root_node

    // Find all nodes that share edges with the current node
    var connected_nodes = adjacency_dict(root_node)

    // Recursively explore all connected nodes
    while (connected_nodes.nonEmpty) {

      // Update the list of used nodes to include the ones from the new iteration
      community ++= connected_nodes

      // For all current nodes, get their children
      var iteration_nodes: mutable.Set[String] = mutable.Set.empty[String]
      for (node <- connected_nodes) {

        // Get the next set of nodes to explore
        var connected_nodes_next = adjacency_dict(node)
        iteration_nodes ++= connected_nodes_next
      }
      // Find the next set of yet unseen connected nodes
      connected_nodes = iteration_nodes.diff(community)
    }
    // Handle unconnected nodes
    //if (community.isEmpty) {
    //  community += root_node}

    return community
  }



  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()

    // Get user inputs
    val filter_threshold = args(0).toInt
    val input_file_path = args(1)
    val betweenness_output_file_path = args(2)
    val community_output_file_path = args(3)

    // Initialize Spark with the 4 GB memory parameters from HW1
    val config = new SparkConf().setMaster("local[*]").setAppName("task2").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")//.set("spark.testing.memory", "471859200")
    val sc = SparkContext.getOrCreate(config)
    sc.setLogLevel("WARN")

    // Read the CSV skipping its header, and reshape it as (user_a, user_b)
    val inputRDDwithHeader = sc.textFile(input_file_path)
    val inputHeader = inputRDDwithHeader.first()
    val inputRDD = inputRDDwithHeader.filter(row => row != inputHeader).map(row => (row.split(',')(0), row.split(',')(1))).groupByKey().mapValues(x => x.toSet).collect()

    // Convert users to nodes and create edges between those whose shared business cross the threshold
    val nodes: mutable.Set[String] = mutable.Set.empty[String]
    val edges: mutable.Set[(String, String)] = mutable.Set.empty[(String, String)]
    for (user_a <- inputRDD){
      for (user_b <- inputRDD){
        //Ensure we don't compare an user against itself
        if (user_a._1 != user_b._1){
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

    // For each node, get a set of all other nodes it connects to (it has edges with)
    val adjacency_dict: mutable.Map[String, mutable.Set[String]] = mutable.Map.empty[String, mutable.Set[String]]
    for ((left_node, right_node) <- edges) {
      if (!adjacency_dict.contains(left_node)) {
        adjacency_dict += (left_node -> mutable.Set(right_node))
      }
      else {
      adjacency_dict(left_node).add(right_node)
      }
    }

    //	Apply the Girvan Newman algorithm and find communities
    var	communities = sc.parallelize(nodes.toSeq).map(n => GN(n, adjacency_dict, nodes))

    // 	From the communities, calculate betweennees of nodes (dividing by two to accout for the two-way road)
    var B_score = communities.flatMap(x => x).reduceByKey(_ + _).map(row => (row._1, row._2 / 2)).sortBy(row => (-row._2, row._1)).collect()

    // Save the pairs and thei betweenness to a txt
    var pw = new PrintWriter(new File(betweenness_output_file_path))
    for (((left_node, right_node), betweenness) <- B_score)
      pw.write("('" + left_node + "', '" + right_node + "')," + (betweenness * 100000).round / 100000.toDouble + "\n")
    pw.close()

    // Calculate the degree matrix
    val DM = mutable.Map.empty[String, Int]
    for ((node, connected_nodes) <- adjacency_dict) {
        DM += (node -> connected_nodes.size)
    }

    // Calculate the adjacency matrix
    val AM = mutable.Map.empty[(String, String), Int]
    for (left_node <- nodes){
      for (right_node <- nodes){
        if (edges.contains((left_node, right_node))) {
          AM += ((left_node, right_node) -> 1)
        }
        else {
          AM += ((left_node, right_node) -> 0)
        }
      }
    }

    // Calculate m (number of edges in the original graph)
    val m = edges.size.toDouble / 2

    // Track the number of edges remaning
    var remaning_edges = m

    // Track the modularity
    var Q = Double.NegativeInfinity

    // Track best communities (that with highest modularity)
    var communities2: mutable.ListBuffer[mutable.Set[String]] = mutable.ListBuffer.empty[mutable.Set[String]]

    // Iteratively remove edges and track the changes in modularity
    while (remaning_edges > 0){

      // Apply the Girvan Newman algorithm and find communities in the ever-shrinking adjacency dict
      var new_graph = sc.parallelize(nodes.toSeq).map(n => GN(n, adjacency_dict, nodes))

      // Recompute betweenness to be used in this iteration
      var B_score = new_graph.flatMap(x => x).reduceByKey(_ + _).map(row => (row._1, row._2 / 2)).sortBy(row => (-row._2, row._1)).collect()

      // Take highest "yet unremoved" betweenness
      var curr_betweenness = B_score(0)._2

      // Then remove all pairs whose betweenness matches "curr_betweenness"
      for (((left_node, right_node), betweenness) <- B_score) {
        if (betweenness >= curr_betweenness) {
          adjacency_dict(left_node).remove(right_node)
          adjacency_dict(right_node).remove(left_node)
          remaning_edges -= 1
        }
      }

      // Recompute communities taking into consideration the removed links
      var iter_communities = compute_communities(adjacency_dict, nodes)

      // Recompute the modularity score Q
      var iter_Q = 0.toDouble
      for (comm <- iter_communities) {
        for (left_node <- comm) {
          for (right_node <- comm) {
            iter_Q += (AM((left_node, right_node)) - DM(left_node) * DM(right_node) / (2 * m))
          }
        }
      }

      // If the modularity score improved, store the current Q and communities as the best thus far
      if (iter_Q > Q) {
        communities2 = iter_communities
        Q = iter_Q
      }
    }

    // Once all edged have been explores, take the sort the best communities found (necessary for proper outputting)
    val output_communities = sc.parallelize(communities2).map(community => community.toList).sortBy(community => (community.size, community.head)).collect()

    // Save the communities to a txt
    pw = new PrintWriter(new File(community_output_file_path))
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