from flask import Flask, request, jsonify, Response
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import math
import json
import os

app = Flask(__name__)

class Graph:
    def __init__(self):
        self.graph = {}

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = {}

    def add_edge(self, node1, node2, weight):
        if node1 in self.graph and node2 in self.graph:
            self.graph[node1][node2] = weight
            self.graph[node2][node1] = weight

    def remove_node(self, node):
        if node in self.graph:
            del self.graph[node]
            for n in self.graph:
                if node in self.graph[n]:
                    del self.graph[n][node]

    def get_networkx_graph(self):
        G = nx.Graph()
        for node, edges in self.graph.items():
            G.add_node(node)
            for neighbor, weight in edges.items():
                G.add_edge(node, neighbor, weight=weight)
        return G

def dijkstra_shortest_path(graph, start, end):
    visited = set()
    distances = {node: math.inf for node in graph}
    previous_nodes = {node: None for node in graph}
    distances[start] = 0

    while distances:
        current_node = min(distances, key=lambda node: distances[node])
        if distances[current_node] == math.inf:
            break

        for neighbor, weight in graph[current_node].items():
            if neighbor in visited:
                continue
            new_distance = distances[current_node] + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current_node

        del distances[current_node]
        visited.add(current_node)

    path = []
    while end:
        path.insert(0, end)
        end = previous_nodes[end]

    return path

graph = Graph()

@app.route("/")
def hello_world():
    return "<p>¡Está funcionando bien! :D</p>"

@app.route('/add_node', methods=['POST'])
def add_node():
    data = request.get_json()
    node = data.get('node')
    graph.add_node(node)
    return jsonify({"message": f"Node {node} added to the graph"})

@app.route('/add_edge', methods=['POST'])
def add_edge():
    data = request.get_json()
    node1 = data.get('node1')
    node2 = data.get('node2')
    weight = data.get('weight')
    graph.add_edge(node1, node2, weight)
    return jsonify({"message": f"Edge added between {node1} and {node2} with weight {weight}"})

@app.route('/remove_node', methods=['DELETE'])
def remove_node():
    data = request.get_json()
    node = data.get('node')
    graph.remove_node(node)
    return jsonify({"message": f"Node {node} removed from the graph"})

@app.route('/show_graph', methods=['GET'])
def show_graph():
    G = graph.get_networkx_graph()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, font_size=10, font_color="black", node_color="skyblue", font_weight="bold", width=2)

    if 'shortest_path' in request.args:
        try:
            start_node = request.args['start_node']
            end_node = request.args['end_node']
            path = dijkstra_shortest_path(graph.graph, start_node, end_node)
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            edge_colors = ['r' if edge in path_edges else 'k' for edge in G.edges()]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)
        except Exception:
            pass

    img = BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    return Response(img, mimetype="image/png")

@app.route('/shortest_path', methods=['POST'])
def shortest_path():
    data = request.get_json()
    start_node = data.get('start_node')
    end_node = data.get('end_node')

    if start_node not in graph.graph or end_node not in graph.graph:
        return jsonify({"message": "Start or end node not found"})

    try:
        path = dijkstra_shortest_path(graph.graph, start_node, end_node)
        return jsonify({"shortest_path": path})
    except Exception:
        return jsonify({"message": "No route found"})

@app.route('/load_data', methods=['GET'])
def load_data():
    try:
        # Cargar nodos desde el archivo "paradas.json"
        with open(os.path.join('data', 'paradas.json'), 'r') as nodes_file:
            nodes_data = json.load(nodes_file)

        # Agregar nodos al grafo
        for node_data in nodes_data:
            node = node_data['name']
            graph.add_node(node)

        # Cargar aristas desde el archivo "distancias.json"
        with open(os.path.join('data', 'distancias.json'), 'r') as edges_file:
            edges_data = json.load(edges_file)

        # Agregar aristas al grafo
        for edge_data in edges_data:
            origin_node = edge_data['origen']
            destination_node = edge_data['destino']
            weight = int(edge_data['kilometros'])
            graph.add_edge(origin_node, destination_node, weight)

        return jsonify({"message": "Datos cargados correctamente al grafo."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)