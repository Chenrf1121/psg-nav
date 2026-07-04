import numpy as np

class RoomNode():
    def __init__(self, caption):
        self.caption = caption
        self.nodes = set()
        self.group_nodes = []


class GroupNode():
    def __init__(self, caption=''):
        self.caption = caption
        self.center = None
        self.center_node = None
        self.nodes = []
        self.edges = set()
        self.probability = 1.0  # Probability of this caption combination

    def __lt__(self, other):
        return self.probability < other.probability

    def get_graph(self):
        self.center = np.array([node.center for node in self.nodes]).mean(axis=0)
        min_distance = np.inf
        for node in self.nodes:
            distance = np.linalg.norm(np.array(node.center) - np.array(self.center))
            if distance < min_distance:
                min_distance = distance
                self.center_node = node
            self.edges.update(node.edges)
        self.caption = self.graph_to_text(self.nodes, self.edges)

    def graph_to_text(self, nodes, edges):
        nodes_text = ', '.join([node.caption for node in nodes])
        edges_text = ', '.join([f"{edge.node1.caption} {edge.relation} {edge.node2.caption}" for edge in edges])
        return f"Nodes: {nodes_text}. Edges: {edges_text}."

    def copy(self):
        """Create a deep copy of this GroupNode."""
        new_group = GroupNode(caption=self.caption)
        new_group.center = self.center.copy() if self.center is not None else None
        new_group.center_node = self.center_node  # Reference, not copy
        new_group.nodes = self.nodes.copy()  # Shallow copy of list
        new_group.edges = self.edges.copy()  # Shallow copy of set
        new_group.probability = self.probability
        return new_group


class ObjectNode():
    def __init__(self):
        self.is_new_node = True
        self.is_goal_node = False
        self.caption = None
        self.object = None
        self.center = None
        self.room_node = None
        self.edges = set()

    def add_edge(self, edge):
        self.edges.add(edge)

    def remove_edge(self, edge):
        self.edges.discard(edge)
    
    def set_caption(self, new_caption):
        for edge in list(self.edges):
            edge.delete()
        self.is_new_node = True
        self.caption = new_caption
        self.edges.clear()
    
    def set_object(self, object):
        self.object = object
        self.object['node'] = self
    
    def set_center(self, center):
        self.center = center
        
    def copy(self):
        new_node = ObjectNode()
        new_node.caption = self.caption
        new_node.object = self.object.copy() if hasattr(self.object, 'copy') else self.object
        new_node.center = self.center
        new_node.room_node = self.room_node
        new_node.is_goal_node = self.is_goal_node
        new_node.edges = self.edges.copy() if hasattr(self.edges, 'copy') else self.edges
        return new_node