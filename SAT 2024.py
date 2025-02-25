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
    

  def get_dist(self,place1,place2):
    # Returns the distance between two place names (strings) if an edge exists,
    # otherwise returns -1.
  
    for edge in self.edges:
      if edge.place1 == place1 and edge.place2 == place2:
        return edge.dist
      if edge.place1 == place2 and edge.place2 == place1:
        return edge.dist
    return -1

  
  def display(self, filename = "map.png"):
    # Displays the object on screen and also saves it to a PNG named in the argument.
    
    edge_labels = {}
    edge_colours = []
    G = nx.Graph()
    node_colour_list = []
    for node in self.nodes:
      G.add_node(node.name, pos=(node.long, node.lat))
      node_colour_list.append(self.colour_dict[node.colour])
    for edge in self.edges:
      G.add_edge(edge.place1, edge.place2)
      edge_labels[(edge.place1, edge.place2)] = edge.dist
      edge_colours.append(self.colour_dict[edge.colour])
    node_positions = nx.get_node_attributes(G, 'pos')


    plt.figure(figsize=(10, 8))
    nx.draw(G, node_positions, with_labels=True, node_size=50, node_color=node_colour_list, font_size=8, font_color='black', font_weight='bold', edge_color=edge_colours)
    nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_labels)
    plt.title('')
    plt.savefig(filename)
    plt.show()


  def haversine(self, lat1, lon1, lat2, lon2):
  # Returns the distance in km between two places with given latitudes and longitudes.
  
      # Radius of the Earth in kilometers
      R = 6371.0

      # Convert latitude and longitude from degrees to radians
      lat1_rad = math.radians(lat1)
      lon1_rad = math.radians(lon1)
      lat2_rad = math.radians(lat2)
      lon2_rad = math.radians(lon2)

      # Differences in coordinates
      dlat = lat2_rad - lat1_rad
      dlon = lon2_rad - lon1_rad

      # Haversine formula
      a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
      c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

      # Calculate the distance
      distance = R * c

      return distance
  
  # This is where you will write your algorithms. You don't have to use
  # these names/parameters but they will probably steer you in the right
  # direction.

# =============================find the shortest path from bendigo==========================
 
  def findpath(self, start, end):
    distances = {node.name: float('inf') for node in self.nodes} # setting all node distances to infinitely 
    distances[start] = 0 # setting the start node to a distance of 0
    times = {node.name: 0 for node in self.nodes}  # New dictionary for tracking time all nodes Initially time is 0

    unvisited_vertices = [node.name for node in self.nodes] # a list of nodes not yet visted Initially this includes all nodes.

    predecessors = {} #Dictionary which will map each node to its predecessor on the shortest path from the start node

    while unvisited_vertices: # Loop will run till there are no more unvisted vertices
        min_distance_vertex = min(unvisited_vertices, key=lambda v: distances[v]) # Finds node with smallest distance from start node assigning it to min_distance_vertex
        if distances[min_distance_vertex] == float('inf'):
            break  # if the smallest distance is infinity, no further paths are accessible
        unvisited_vertices.remove(min_distance_vertex)

        for node in self.nodes: # finds min_distance_vertex node in the object so it can access neighbours
            if node.name == min_distance_vertex:
                current_node_obj = node
                break

        for neighbor_name in current_node_obj.neighbours: # loops over the neighbors names of current node
            #  get the edge distance and time between the current node and the neighbor
            edge_distance, edge_time = self.get_edge_details(min_distance_vertex, neighbor_name)
            # calculate new distance and time to the neighbor
            new_distance = distances[min_distance_vertex] + edge_distance
            new_time = times[min_distance_vertex] + edge_time

            # Update if a shorter path found
            if new_distance < distances[neighbor_name]:
                distances[neighbor_name] = new_distance
                times[neighbor_name] = new_time
                predecessors[neighbor_name] = min_distance_vertex

        if min_distance_vertex == end: # checks if the current node is the destination if so returns the path, total distance and timew
            path = []
            current_node = end
            total_distance = distances[end]
            total_time = times[end]
            while current_node is not None: # creates the path to return recreating path frm end node to start node by following each node's predecessor.
                path.append(current_node)
                current_node = predecessors.get(current_node)
            return path[::-1], total_distance, total_time

    return None

  def get_edge_details(self, place1, place2):
    for edge in self.edges:
        if (edge.place1 == place1 and edge.place2 == place2) or (edge.place1 == place2 and edge.place2 == place1):
            return edge.dist, edge.time
    return float('inf'), 0  # In case there is no direct edge between place1 and place2



# ==========================find shortest path to all towns in radius==============================

  def townInRadius(self, place,radius):
      # Returns the towns within a given radius from another Town
      towns_in_radius = []
      # Checks which town's name matches the one provided by the user.
      # If it matches one it is set as the starting place for the radius to be drawn around.
      for node in self.nodes:
        if node.name == place:
           lat = node.lat
           lon = node.long
      for node in self.nodes: 
          distance = self.haversine(lat, lon, node.lat, node.long)  
          if distance <= radius:
              towns_in_radius.append(node.name)

      return towns_in_radius
  
  # visit every town within a given radius and vaccinate everybody in the shortest possible time.
  def vaccinate(self, towns):
        # Create a subgraph with only towns within the specified radius
        subgraph = Graph()
        subgraph.nodes = [node for node in self.nodes if node.name in towns]
        subgraph.edges = [edge for edge in self.edges if edge.place1 in towns and edge.place2 in towns]

        # Initialize dictionaries to store minimum distances and times using the Floyd-Warshall algorithm
        distances = {}  # Dictionary to store the minimum distances between nodes
        times = {}  # Dictionary to store the minimum travel times between nodes

        # Iterate over all nodes in the subgraph to initialize the dictionaries
        for node1 in subgraph.nodes:
            distances[node1.name] = {}  # Create a nested dictionary for distances from node1
            times[node1.name] = {}  # Create a nested dictionary for times from node1

            for node2 in subgraph.nodes:
                if node1 == node2:
                    # The distance and time from a node to itself is alwaysa 0
                    distances[node1.name][node2.name] = 0
                    times[node1.name][node2.name] = 0
                else:
                    # Initialize the distance and time between different nodes to infinity
                    # we will update this later with actual distances and times
                    distances[node1.name][node2.name] = float('inf')
                    times[node1.name][node2.name] = float('inf')


        # Update distances and times based on existing edges
        # Initialize distances and times based on the edges in the subgraph
        for edge in subgraph.edges:
            distances[edge.place1][edge.place2] = edge.dist  # Set distance from place1 to place2
            distances[edge.place2][edge.place1] = edge.dist  # Set distance from place2 to place1 (undirected graph)
            times[edge.place1][edge.place2] = edge.time  # Set travel time from place1 to place2
            times[edge.place2][edge.place1] = edge.time  # Set travel time from place2 to place1 (undirected graph)


        # Apply Floyd-Warshall algorithm
        for k in subgraph.nodes:
            for i in subgraph.nodes:
                for j in subgraph.nodes:
                    distances[i.name][j.name] = min(distances[i.name][j.name], distances[i.name][k.name] + distances[k.name][j.name])
                    times[i.name][j.name] = min(times[i.name][j.name], times[i.name][k.name] + times[k.name][j.name])

        # construct complete graph based on minimum distances and times
        complete_graph = nx.Graph()
        for node1 in subgraph.nodes:
            for node2 in subgraph.nodes:
                if node1 != node2:
                    complete_graph.add_edge(node1.name, node2.name, weight=distances[node1.name][node2.name])

        # generate all permutations of the nodes
        vertices = list(complete_graph.nodes)
        permutations = itertools.permutations(vertices)

        # Initialize variables to store the shortest path, distance, and time
        shortest_path = None
        shortest_distance = float('inf')
        shortest_time = float('inf')

        # Iterate through all permutations
        for perm in permutations:
            # Calculate the total distance and time for the current permutation
            total_distance = 0
            total_time = 0
            for i in range(len(perm) - 1):
                total_distance += distances[perm[i]][perm[i + 1]]
                total_time += times[perm[i]][perm[i + 1]]

            # update shortest path, distance, and time if current path is shorter
            if total_distance < shortest_distance:
                shortest_distance = total_distance
                shortest_time = total_time
                shortest_path = perm

        return shortest_path, shortest_distance, shortest_time

# =========================Returns results==================================

# Create a new graph object called 'original'
original = Graph()

# Load data into that object.
original.load_data()

# Calls find path function to find distance from Bendigo to any town
start_location = "Bendigo" #Select a start location (Bendigo is the location of the Responce team base so no need to change This)
end_location = "Melbourne" #End location (select the town that needs vaccination)
result = original.findpath(start_location, end_location)

# Checks if a path was returned
if result:
    shortest_path, total_distance, total_time = result
    print(f"Shortest path from {start_location} to {end_location}: {shortest_path}")
    print(f"Total distance: {total_distance} km")
    print(f"Total time: {total_time} mins")
# Otherwise states that no path was found between the two locations 
else:
    print(f"No path found between {start_location} and {end_location}")

# Prints the shortest path to visit all towns in a radius
place = "Melbourne"   #Adjust for any location on Map
radius = 80        #Adjust for any radius size from the town above
result = original.townInRadius(place, radius)
print(f"Towns within the radius of {place}:", result)

# Prints the shortest path for the medical team to take to vist all the towns in a given radius, with the distance and time it will take
path, distance, time = original.vaccinate(result)
print(f"The shortest path is {path} with a distance of {distance} km and time of {time} mins")