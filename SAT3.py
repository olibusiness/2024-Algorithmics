import networkx as nx
import matplotlib.pyplot as plt
import csv
import math
import random

class Node:
    def __init__(self, name, pop, income, lat, long):
        self.name = name
        self.lat = lat
        self.long = long
        self.pop = pop
        self.income = income
        self.colour = 1
        self.neighbours = []

    def add_neighbour(self, neighbour):
        if neighbour not in self.neighbours:
            self.neighbours.append(neighbour)

class Edge:
    def __init__(self, place1, place2, dist, time):
        self.place1 = place1
        self.place2 = place2
        self.dist = dist
        self.time = time
        self.colour = 2

class Graph:
    def __init__(self):
        self.edges = []
        self.nodes = []
        self.colour_dict = {0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow", 5: "lightblue"}

    def load_data(self):
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

        with open("edges.csv", "r", encoding='utf-8-sig') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                place1 = row[0]
                place2 = row[1]
                dist = int(row[2])
                time = int(row[3])
                edge = Edge(place1, place2, dist, time)
                self.edges.append(edge)

                for node in self.nodes:
                    if node.name == place1:
                        node.add_neighbour(place2)
                    if node.name == place2:
                        node.add_neighbour(place1)

    def display(self, filename="map.png"):
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
        R = 6371.0
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def townInRadius(self, place, radius):
        # Returns the towns within a given radius from another Town
        towns_in_radius = []
        # Checks which town's name matches the one provided by the user.
        # If it matches one it is set as the starting place for the radius to be drawn around.
        for node in self.nodes:
            if node.name == place:
                lat = node.lat
                lon = node.long
        # uses haversine function to calculate distance in a radius around starting town
        for node in self.nodes:
            distance = self.haversine(lat, lon, node.lat, node.long)
            if distance <= radius:
                towns_in_radius.append(node.name)
        return towns_in_radius

    def vaccinate(self, towns):
        # Create a subgraph with only towns within the specified radius
        subgraph = nx.Graph()
        
        # Add nodes and edges to the subgraph
        for node in self.nodes:
            if node.name in towns:
                subgraph.add_node(node.name, pos=(node.long, node.lat))
        
        for edge in self.edges:
            if edge.place1 in towns and edge.place2 in towns:
                subgraph.add_edge(edge.place1, edge.place2, weight=edge.dist, time=edge.time)

        # initialize dictionaries to store minimum distances and time
        distances = {}
        times = {}

        # set initial distances to infinity
        for node in subgraph.nodes:
            distances[node] = {}
            times[node] = {}
    
            for node2 in subgraph.nodes:
                distances[node][node2] = float('inf')
                times[node][node2] = float('inf')
        
        # set self-distances and self-times to 0
        for node in subgraph.nodes:
            distances[node][node] = 0
            times[node][node] = 0

        # loops over each edge in the subgraph to set distances and times
        for u, v, data in subgraph.edges(data=True):
            # Set the distance from node u to node v using the weight attribute from the edge data.
            distances[u][v] = data['weight']
            # since the graph is undirected, also set the distance from node v to node u.
            distances[v][u] = data['weight']
            # Set the travel time from node u to node v using the time attribute from the edge data.
            times[u][v] = data['time']
            # again, set the travel time from node v to node u for the undirected graph.
            times[v][u] = data['time']


        # Apply Floyd-Warshall algorithm to compute shortest paths
        for k in subgraph.nodes:
            for i in subgraph.nodes:
                for j in subgraph.nodes:
                    distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])
                    times[i][j] = min(times[i][j], times[i][k] + times[k][j])

        # Create a complete graph with minimum distances
        complete_graph = nx.Graph()
        for node1 in subgraph.nodes:
            for node2 in subgraph.nodes:
                if node1 != node2:
                    complete_graph.add_edge(node1, node2, weight=distances[node1][node2])

        # initialize counters
        path_mutations = 0
        comparisons = 0

        def get_distance(path):
            # Calculate the total distance of a path

            total_distance = 0  # initialize total distance to 0
            
            # loop through the path, ading the distances of each edge
            # the loop runs from the first node to the second to last node in the path
            # this is because each edge is defiend by a pair of consecutive nodes 
            # except the last node which doesn't have a subsequent node to form a pair
            for i in range(len(path) - 1):
                u = path[i]  # Current node
                v = path[i + 1]  # Next node
                total_distance += complete_graph[u][v]['weight']  # add the distance between nodes u and v to the total 

            return total_distance  # Return the total distance of the path

        def get_time(path):
            # calculates the total travel time for a path

            total_time = 0  # Initialize total time to 0

            # loop through the path, adding the travel times of each segment
            # Same as in the get_distance function
            for i in range(len(path) - 1):
                u = path[i]  # current node
                v = path[i + 1]  # Next node
                total_time += times[u][v]  # add the travel time between nodes u and v to total time

            return total_time  # Return the total travel time of the path


        def town_mutation(path):
            nonlocal path_mutations
            path_mutations += 1
            # Inversion Mutation
            path = path[:]  # Create a copy of the path to avoid modifying the original list
            i, j = random.sample(range(len(path)), 2)  # randomly selct two distinct indices i and j from the path
            path[i:j+1] = reversed(path[i:j+1])  # reverse the sublist of path from index i to j (inclusive)
            return path  # return the modified path

        
        def calculate_cooling_rate(k, T0, Tmin):
            # Parameters
            # k: The desired number of temperature steps
            # T0: Initial temperature (default is 100)
            # Tmin: Minimum temperature (default is 0.1)
        
            # Calculate the cooling rate using the formula (check SAT 3)
            alpha = math.exp(math.log(Tmin / T0) / k)
            return alpha #  returns the cooling rate alpha


        def simulated_annealing(initial_path):
                # accesses nonlocal variables
                nonlocal path_mutations, comparisons

                # initialize the curent path and its cost
                current_path = initial_path
                current_cost = get_distance(current_path)
                
                # Initialize the best path and cost set to the first path initialy
                best_path = current_path
                best_cost = current_cost
                
                # initial temperture setting for the annealing process
                temperature = 100  # starting temperature
                # Changes number of attempted solutions. Edit this to change the number of solutions (controls the cooling rate)
                k = 100000

                # the minimum accepted Temp  
                Tmin = 0.1
                # Calculate the cooling rate based on the desired number of attempted solutions
                cooling_rate = calculate_cooling_rate(k, temperature, Tmin)

                # main loop for the simulated annealing process
                while temperature > Tmin:  # continue until temperature is low enough 
                    # Generate a new path and get its cost (by mutating the current path)
                    new_path = town_mutation(current_path)
                    new_cost = get_distance(new_path)
                    
                    # increment the comparison count
                    comparisons += 1
                    
                    # decide wether to accept the new path based on cost and probability
                    if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
                        # Update the current path and cost
                        current_path = new_path
                        current_cost = new_cost
                        
                        # if the solution is the best update the best path and cost
                        if new_cost < best_cost:
                            best_path = new_path
                            best_cost = new_cost
                    
                    # Cool down the temperature (with the calculated cooling rate)
                    temperature *= cooling_rate

                # Return the best path and its cost
                return best_path, best_cost

        # Generate an initial path (a random permutation of nodes)
        initial_path = list(complete_graph.nodes)
        random.shuffle(initial_path)

        # apply simulated annealing to find the best path
        best_path, best_cost = simulated_annealing(initial_path)
        best_time = get_time(best_path)
        
        # Print counters for analysis (may be 1 above due to rounding)
        # print(f"Path mutations: {path_mutations}")
        print(f"attempted solutions: {comparisons}")
        
        return best_path, best_cost, best_time

# Create a new graph object called 'original'
original = Graph()

# Load data into that object.
original.load_data()

# Prints the shortest path to visit all towns in a radius
place = "Melbourne"
radius = 150
result = original.townInRadius(place, radius)

# Call the vaccinate method and unpack the returned values
path, distance, time = original.vaccinate(result)
print(f"Shortest path is {path} with a distance of {distance}km and time of {time} mins")

