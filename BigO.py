import networkx as nx
import matplotlib.pyplot as plt
import csv
import math
import itertools

class Node:
    def __init__(self, name, pop, income, lat, long):
        # Name, latitude, longitude, population, weekly household income, default colour (1-5), empty list of neighbours
        self.name = name
        self.lat = lat
        self.long = long
        self.pop = pop
        self.income = income
        self.colour = 1
        self.neighbours = []

   
        
    def add_neighbour(self, neighbour):
      # Adds a neighbour (node object) after checking to see if it was there already
      if neighbour not in self.neighbours:
        self.neighbours.append(neighbour)
        

class Edge:
  def __init__(self, place1, place2, dist, time):
    # Two places (order unimportant), distance in km, time in mins, default colour (1-5)
    self.place1 = place1
    self.place2 = place2
    self.dist = dist
    self.time = time
    self.colour = 2

class Graph:
  def __init__(self):
    # List of edge objects and node objects
    self.edges = []
    self.nodes = []
    self.colour_dict = {0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow", 5:"lightblue"}

  def load_data(self):
      # Reads the CSV files you are provided with and creates node/edges accordingly.
      # You should not need to change this function.

      # Read the nodes, create node objects and add them to the node list.
      with open("nodes.csv", 'r', encoding='utf-8-sig') as csvfile:
          csv_reader = csv.reader(csvfile)
          for row in csv_reader:
              name = row[0]
              pop = row[1]
              income = row[2]
              lat = float(row[3])
              long = float(row[4])
              node = Node(name, pop, income, lat, long)
              self.nodes.append(node)

      # Read the edges, create edge objects and add them to the edge list.
      with open("edges.csv", "r", encoding='utf-8-sig') as csvfile:
          csv_reader = csv.reader(csvfile)
          for row in csv_reader:
              place1 = row[0]
              place2 = row[1]
              dist = int(row[2])
              time = int(row[3])
              edge = Edge(place1, place2, dist, time)
              self.edges.append(edge)

              # Add neighbors to the nodes
              for node in self.nodes:
                  if node.name == place1:
                      node.add_neighbour(place2)
                  if node.name == place2:
                      node.add_neighbour(place1)



     
def vaccinate(self, towns):
        # Create a subgraph with only towns within the specified radius
        subgraph = Graph()
        subgraph.nodes = [node for node in self.nodes if node.name in towns] #For each node so this is O(n)
        subgraph.edges = [edge for edge in self.edges if edge.place1 in towns and edge.place2 in towns] #O(e)
        # this results in O(n+e)

        distances = {}  #O(1)
        times = {}  #O(1)

        # Nested for loops which leads to O(n^2)
        for node1 in subgraph.nodes:
            distances[node1.name] = {} #O(1)
            times[node1.name] = {} #O(1)

            for node2 in subgraph.nodes:
                if node1 == node2:
                    # The distance and time from a node to itself is alwaysa 0
                    distances[node1.name][node2.name] = 0
                    times[node1.name][node2.name] = 0
                else:
                    distances[node1.name][node2.name] = float('inf')
                    times[node1.name][node2.name] = float('inf')


        # Loops over all edges so O(E)
        for edge in subgraph.edges:
            distances[edge.place1][edge.place2] = edge.dist  # Set distance from place1 to place2
            distances[edge.place2][edge.place1] = edge.dist  # Set distance from place2 to place1 (undirected graph)
            times[edge.place1][edge.place2] = edge.time  # Set travel time from place1 to place2
            times[edge.place2][edge.place1] = edge.time  # Set travel time from place2 to place1 (undirected graph)


        #Floyd-Warshall algorithm has 3 nested for loops which leads to O(n^3)
        for k in subgraph.nodes:
            for i in subgraph.nodes:
                for j in subgraph.nodes:
                    distances[i.name][j.name] = min(distances[i.name][j.name], distances[i.name][k.name] + distances[k.name][j.name])
                    times[i.name][j.name] = min(times[i.name][j.name], times[i.name][k.name] + times[k.name][j.name])

        # construct complete graph based on minimum distances and times
        # this reults in a O(n^2)
        complete_graph = nx.Graph()
        for node1 in subgraph.nodes: #O(n)
            for node2 in subgraph.nodes: #O(n)
                if node1 != node2: #O(1)
                    complete_graph.add_edge(node1.name, node2.name, weight=distances[node1.name][node2.name]) #O(1)

        # generate all permutations of the nodes
        vertices = list(complete_graph.nodes)
        permutations = itertools.permutations(vertices) 
        # permutations is O(n!)

        # Initialize variables to store the shortest path, distance, and time
        shortest_path = None #O(1)
        shortest_distance = float('inf') #O(1)
        shortest_time = float('inf') #O(1)

        # Iterate through all permutations
        
        for perm in permutations: #O(n!)
            total_distance = 0  #O(1)
            total_time = 0  #O(1)
            for i in range(len(perm) - 1): #O(n)
                total_distance += distances[perm[i]][perm[i + 1]] #O(1)
                total_time += times[perm[i]][perm[i + 1]] #O(1)
                #so in total we would have O(n!) * O(n) = O(n!*n)

            # update shortest path, distance, and time if current path is shorter
            if total_distance < shortest_distance:#O(1)
                shortest_distance = total_distance#O(1)
                shortest_time = total_time#O(1)
                shortest_path = perm#O(1)

        return shortest_path, shortest_distance, shortest_time #O(1)