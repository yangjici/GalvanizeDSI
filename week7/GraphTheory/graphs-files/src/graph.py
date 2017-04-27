class Vertex(object):
    def __init__(self, name):
        self.name = name
        self.neighbors = {}

    def add_neighbor(self, neighbor, weight):
        self.neighbors[neighbor] = weight


class Graph(object):
    def __init__(self):
        self.vertices = {}

    def add_node(self, name):
        if name not in self.vertices:
            self.vertices[name] = Vertex(name)

    def add_edge(self, a, b, weight):
        self.add_node(a)
        self.add_node(b)
        self.vertices[a].add_neighbor(b, weight)
        self.vertices[b].add_neighbor(a, weight)

    def get_neighbors(self, node):
        if node in self.vertices:
            return self.vertices[node].neighbors
        return []


def create_graph():
    g = Graph()
    g.add_edge('sunset', 'richmond', 4)
    g.add_edge('presidio', 'richmond', 1)
    g.add_edge('pac heights', 'richmond', 8)
    g.add_edge('western addition', 'richmond', 7)
    g.add_edge('western addition', 'pac heights', 2)
    g.add_edge('western addition', 'downtown', 3)
    g.add_edge('western addition', 'haight', 4)
    g.add_edge('mission', 'haight', 1)
    g.add_edge('mission', 'soma', 5)
    g.add_edge('downtown', 'soma', 5)
    g.add_edge('downtown', 'nob hill', 2)
    g.add_edge('marina', 'pac heights', 2)
    g.add_edge('marina', 'presidio', 4)
    g.add_edge('marina', 'russian hill', 3)
    g.add_edge('nob hill', 'russian hill', 1)
    g.add_edge('north beach', 'russian hill', 1)
    return g
