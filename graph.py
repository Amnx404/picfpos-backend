
#check:
# Completing the edges with distances and labels

import networkx as nx

# Define the edges with their distances and labels
edges_with_distances_and_labels = {
    ('WM1', 'WM2'): {'distance': 9, 'label': 90},
    ('WM2', 'WM1'): {'distance': 9, 'label': 270},
    ('EX', 'C0'): {'distance': 6, 'label': 90},
    ('C0', 'EX'): {'distance': 6, 'label': 270},
    ('C0', 'C5'): {'distance': 5, 'label': 90},
    ('C5', 'C0'): {'distance': 5, 'label': 270},
    ('C1', 'C2'): {'distance': 7, 'label': 90},
    ('C2', 'C1'): {'distance': 7, 'label': 270},
    ('C5', 'SV'): {'distance': 3, 'label': 90},
    ('SV', 'C5'): {'distance': 3, 'label': 270},
    ('E1', 'E2'): {'distance': 7, 'label': 90},
    ('E2', 'E1'): {'distance': 7, 'label': 270},
    ('E2', 'M'): {'distance': 6, 'label': 90},
    ('M', 'E2'): {'distance': 6, 'label': 270},
    ('I', 'C6'): {'distance': 3, 'label': 90},
    ('C6', 'I'): {'distance': 3, 'label': 270},
    ('C2', 'C3'): {'distance': 4, 'label': 90},
    ('C3', 'C2'): {'distance': 4, 'label': 270},
    ('C3', 'C4'): {'distance': 5, 'label': 90},
    ('C4', 'C3'): {'distance': 5, 'label': 270},
    ('EXB', 'C9'): {'distance': 8, 'label': 90},
    ('C9', 'EXB'): {'distance': 8, 'label': 270},
    ('C10', 'ENT'): {'distance': 4, 'label': 90},
    ('ENT', 'C10'): {'distance': 4, 'label': 270},
    ('C7', 'MEET'): {'distance': 4, 'label': 90},
    ('MEET', 'C7'): {'distance': 4, 'label': 270},
    ('R', '3D'): {'distance': 5, 'label': 90},
    ('3D', 'R'): {'distance': 5, 'label': 270},
    ('C0', 'WM2'): {'distance': 8, 'label': 0},
    ('WM2', 'C0'): {'distance': 8, 'label': 180},
    ('MOD', 'C1'): {'distance': 8, 'label': 0},
    ('C1', 'MOD'): {'distance': 8, 'label': 180},
    ('C5', 'E1'): {'distance': 8, 'label': 0},
    ('E1', 'C5'): {'distance': 8, 'label': 180},
    ('C1', 'C5'): {'distance': 4, 'label': 0},
    ('C5', 'C1'): {'distance': 4, 'label': 180},
    ('C8', 'C2'): {'distance': 8, 'label': 0},
    ('C2', 'C8'): {'distance': 8, 'label': 180},
    ('C2', 'C6'): {'distance': 4, 'label': 0},
    ('C6', 'C2'): {'distance': 4, 'label': 180},
    ('C6', 'E2'): {'distance': 8, 'label': 0},
    ('E2', 'C6'): {'distance': 8, 'label': 180},
    ('C3', 'TEC'): {'distance': 4, 'label': 0},
    ('TEC', 'C3'): {'distance': 4, 'label': 180},
    ('R', 'C3'): {'distance': 5, 'label': 0},
    ('C3', 'R'): {'distance': 5, 'label': 180},
    ('C4', 'SEW'): {'distance': 4, 'label': 0},
    ('SEW', 'C4'): {'distance': 4, 'label': 180},
    ('3D', 'C4'): {'distance': 5, 'label': 0},
    ('C4', '3D'): {'distance': 5, 'label': 180},
    ('EXB', '3D'): {'distance': 4, 'label': 0},
    ('3D', 'EXB'): {'distance': 4, 'label': 180},
    ('C7', 'C10'): {'distance': 5, 'label': 0},
    ('C10', 'C7'): {'distance': 5, 'label': 180},
    ('C9', 'C7'): {'distance': 4, 'label': 0},
    ('C7', 'C9'): {'distance': 4, 'label': 180},
    ('C4', 'C10'): {'distance': 3, 'label': 90},
    ('C10', 'C4'): {'distance': 3, 'label': 270}
}


# Create a directed graph
G_directional_distances = nx.DiGraph()

# Add edges to the graph with distances as weights
for edge, attributes in edges_with_distances_and_labels.items():
    G_directional_distances.add_edge(edge[0], edge[1], weight=attributes['distance'], label=attributes['label'])

# Function to find the shortest path with Dijkstra's algorithm considering distance
def find_shortest_path_dijkstra(graph, start_node, end_node):
    try:
        # Compute the shortest path using Dijkstra's algorithm
        path = nx.dijkstra_path(graph, source=start_node, target=end_node, weight='weight')
        for i in range(len(path)-1):
            edge = (path[i], path[i+1])
            direction = graph.edges[edge]['label']
            distance = graph.edges[edge]['weight']
        return path
    except (nx.NetworkXNoPath, KeyError):
        # Return an empty list if no path exists or if the nodes are not in the graph
        return []



def find_shortest_path_bfs(graph, start_node, end_node):

    try:
        path = find_shortest_path_dijkstra(graph, start_node, end_node)
        return path
    except (nx.NetworkXNoPath, KeyError):
        # Return an empty list if no path exists or if the nodes are not in the graph
        return []

# Example usage of the function
# Find the shortest path from 'CAF' to 'ENT'
# path_caf_to_ent = find_shortest_path_bfs(G_directional_distances, 'ENT', 'WM1')
# path_caf_to_ent

shift = 45
def get_direction_and_path(graph, start_node, end_node, alpha_rotation):
    """
    Get the direction and path from start_node to end_node in a graph.

    :param graph: NetworkX graph
    :param start_node: starting node
    :param end_node: ending node
    :param alpha_rotation: current alpha rotation with respect to true North
    :return: tuple (path, directions)
    """

    def calculate_direction(edge, current_alpha):
        """
        Calculate the direction to face for a given edge, adjusting for the current alpha rotation.
        The graph directions are 45 degrees to the right of true north and are clockwise.
        """
        graph_direction = graph.edges[edge]['label']
        adjusted_direction = (graph_direction - current_alpha + 360 + shift) % 360
        return adjusted_direction

    path = find_shortest_path_bfs(graph, start_node, end_node)
    directions = []


    edge = (path[0], path[1])
    direction = calculate_direction(edge, alpha_rotation)


    return path, direction




def get_relative_direction(angle):
    """
    Convert an angle to a relative direction (F, L, B, R).

    :param angle: Angle in degrees, where 0 degrees is the direction you're facing
    :return: A string representing the relative direction ('F', 'L', 'B', 'R')
    """
    if 315 <= angle < 360 or 0 <= angle < 45:
        return 'F'  # Forward
    elif 45 <= angle < 135:
        return 'R'  # Right
    elif 135 <= angle < 225:
        return 'B'  # Backward
    elif 225 <= angle < 315:
        return 'L'  # Left
    else:
        return 'Unknown'  # For unexpected cases




def navigate(start_node, end_node, alpha_rotation):
    """
    Navigate from start_node to end_node, adjusting for the current alpha rotation.

    :param start_node: starting node
    :param end_node: ending node
    :param alpha_rotation: current alpha rotation with respect to true North
    :return: tuple (path, directions)
    """
    path, angle = get_direction_and_path(G_directional_distances, start_node, end_node, alpha_rotation)
    directions = get_relative_direction(angle)
    return path, directions

