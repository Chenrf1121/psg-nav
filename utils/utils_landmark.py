import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage.draw import line
import networkx as nx

def euclidean_distance(pos, node1, node2):
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def getAngle(G,node1,node2):
    neighbor1 = list(G.neighbors(node1))[0]
    neighbor2 = list(G.neighbors(node2))[0]
    a1 = np.array([G.nodes()[node1]['pos'][0]-G.nodes()[neighbor1]['pos'][0],G.nodes()[node1]['pos'][1]-G.nodes()[neighbor1]['pos'][1]])
    a2 = np.array([G.nodes()[node2]['pos'][0]-G.nodes()[neighbor2]['pos'][0],G.nodes()[node2]['pos'][1]-G.nodes()[neighbor2]['pos'][1]])
    if np.dot(a1,a2)/(np.linalg.norm(a1)*np.linalg.norm(a2)) > np.cos(np.radians(45)) :
        return True
    else :
        return False
    
def merge_closest_non_leaf_nodes(graph,min_distance = 5):
    non_leaf_nodes = list(nx.get_node_attributes(graph, 'pos').keys())
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    closest_pair = []
    pos = nx.get_node_attributes(graph, 'pos')
    for i, node1 in enumerate(non_leaf_nodes):
        
        for node2 in non_leaf_nodes[i + 1:]:
            distance = euclidean_distance(pos[node1], pos[node2])
            if distance < min_distance and node1 in lengths.keys() and node2 in lengths[node1].keys() and lengths[node1][node2] < 3:
                closest_pair.append((node1, node2)) 
        if closest_pair:
            for pair in closest_pair:
                node1, node2 = pair
                if graph.has_node(node1) and graph.has_node(node2) :
                    # 计算子节点数量
                    num_subnodes1 = graph.degree(node1)

                    # 计算子节点数量
                    num_subnodes2 = graph.degree(node2)
                    if num_subnodes1 >= num_subnodes2 :
                        graph = nx.contracted_nodes(graph, node1, node2, self_loops=False)
                    else :
                        graph = nx.contracted_nodes(graph, node2, node1, self_loops=False)

    return graph

def add_new_nodes_with_condition(G, new_nodes):
    # 遍历新节点
    pos = nx.get_node_attributes(G, 'pos') 
    explored_nodes=[]
    n=len(pos)
    for new_coord in new_nodes:
        # 计算新节点与图中所有现有节点的最小距离
        min_distance = float('inf')
        closest_node = None
        
        for existing_node in G.nodes(data=True):
            existing_coord = existing_node[1]['pos']
            distance = euclidean_distance(new_coord, existing_coord)
            
            if distance < min_distance:
                min_distance = distance
                closest_node = existing_node[0]
        explored_nodes.append(closest_node)
    return G,list(set(explored_nodes))

def CoarseGraph(graph,group,leaf_nodes):
    G = graph.copy()
    pos = nx.get_node_attributes(G, 'pos')
    for grp in group :
        if len(grp) == 2 :
            G.remove_node(grp[1])
            del leaf_nodes[grp[1]]
        elif len(grp) > 2 :
            node_ave=np.array([0,0],dtype=np.float64)
            nodes_array=[]
            neighbors=[]
            for i in grp :
                node_ave += np.array(pos[i])
                nodes_array.append(list(pos[i]))
                neighbors.append(list(graph.neighbors(i))[0])
            node_ave /= len(grp)
            merged_to_node = np.argmin(np.linalg.norm(node_ave-np.array(nodes_array),axis=-1))
            for i,n in zip(grp,neighbors) :
                if i != grp[merged_to_node] :
                    del leaf_nodes[i]
                    G.remove_node(i)
    return G,leaf_nodes
    
def mergeGraphByObjects(graph,value_dict,leaf_position):
    nearest_leaf_nodes={}
    values = np.array(list(value_dict.values()))
    keys = np.array(list(value_dict.keys()))
    value_matrix = (values[:, None] == values[None, :]).astype(int)
    np.fill_diagonal(value_matrix, 0)
    positions= np.array(list(leaf_position.values()))
    distance = np.linalg.norm(positions[None,:,:]-positions[:,None,:],axis=-1)
    judge = ((distance <= 20) *(value_matrix==1))
    judge[np.tril_indices_from(judge, k=-1)] = 0
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    for node in range(len(judge)):
        nearest_leaf = []
        related_node = keys[np.where(judge[node])[0]]
        for node_r in related_node :
            if keys[node] in shortest_path_lengths.keys() and node_r in shortest_path_lengths[keys[node]].keys() and shortest_path_lengths[keys[node]][node_r] <= 5 :
                if getAngle(graph,keys[node],node_r):
                    nearest_leaf.append(node_r)
        nearest_leaf_nodes[keys[node]]=nearest_leaf
    group_dict = {}
    group = []
    for key,values in nearest_leaf_nodes.items():
        if len(values) == 0 :
            length=len(group)
            group.append([key])
            group_dict[key]=length
        else :
            for value in values :
                if key in group_dict.keys():
                    if value not in group_dict.keys() :
                        group_dict[value] = group_dict[key]
                        group[group_dict[key]].append(value)
                else :
                    if value in group_dict.keys():
                        group_dict[key] = group_dict[value]
                        group[group_dict[value]].append(key)
                    else :
                        length=len(group)
                        group.append([key,value])
                        group_dict[key]=length
                        group_dict[value]=length
    return group

def getLeafValue(map_2d,graph):
    leaf_nodes = [node for node in graph.nodes if graph.degree[node] == 1 and 'pos' in graph.nodes[node].keys()]
    node = []
    points_2d=[]
    direction = []
    for i in leaf_nodes:
        points_2d.append([graph.nodes()[i]['pos'][0],graph.nodes()[i]['pos'][1]])
        neighbor =  list(graph.neighbors(i))[0]
        direction.append([graph.nodes()[i]['pos'][0]-graph.nodes()[neighbor]['pos'][0],graph.nodes()[i]['pos'][1]-graph.nodes()[neighbor]['pos'][1]])
        node.append(i)
    
    direction = np.array(direction)
    points_2d = np.array(points_2d)
    direction = direction/np.linalg.norm(direction,axis=-1,keepdims=True)
    scales = np.arange(1,11).reshape(10,1)
    direction = direction[:, np.newaxis, :] * scales
    points_index = direction + points_2d[:,np.newaxis,:]
    indices = np.floor(np.array(points_index)).astype(np.int16)
    indices = np.clip(indices,np.array([0,0]),np.array([map_2d.shape[1]-1,map_2d.shape[0]-1]))
    value = map_2d[indices.reshape(-1,2)[:,1],indices.reshape(-1,2)[:,0]].reshape(-1,10)
    non_zero_one_mask = ((value != 0) * (value != 1))
    first_non_zero_one_indices = np.argmax(non_zero_one_mask, axis=1)
    first_non_zero_one_indices[np.all(~non_zero_one_mask, axis=1)] = -1
    first_non_zero_values = value[np.arange(value.shape[0]), first_non_zero_one_indices]
    leaf_value = first_non_zero_values
    value_dict={}
    value_position={}
    for node,value in zip(leaf_nodes,leaf_value):
            value_dict[node] = value
            value_position[node] = graph.nodes()[node]['pos']
    return value_dict,value_position

def remove_isolated_nodes(G):
    isolated_nodes = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(isolated_nodes)
    return G

def calculate_angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 防止浮点数精度问题
    angle_deg = np.degrees(angle_rad)  # 转换为角度
    return angle_deg

def VorRemoveOut(vor,fea_map,obs_map):
    vertices=vor.vertices
    relation = vor.ridge_vertices
    b_f,b_u = np.min(vor.points,axis=0),np.max(vor.points,axis=0)
    index1 = np.where(((vertices >= b_f) & (vertices < b_u))[:,0] & ((vertices >= b_f) & (vertices < b_u))[:,1])[0]
    verticesfloor= np.round(vertices[index1]).astype(np.int32)
    index_final = np.where(fea_map[verticesfloor[:,1],verticesfloor[:,0]]==1)[0]
    index_remain = index1[index_final]
    v_remain = vertices[index_remain]
    points =np.vstack((np.where(obs_map==1)[1],np.where(obs_map==1)[0])).T
    distance = np.linalg.norm(v_remain[:,None,:] - points[None,:,:],axis=-1)
    index_remain= list(set(index_remain)-set(index_remain[np.where(distance < 4)[0]]))
    vor.vertices = vertices[index_remain]
    relation_new=[]
    for i in range(len(relation)):
        if relation[i][0] in index_remain and relation[i][1] in index_remain and relation[i][0] >=0 and relation[i][1] >=0:
            relation_new.append((index_remain.index(relation[i][0]),index_remain.index(relation[i][1])))
    vor.ridge_vertices = relation_new
    vor.vertices= vertices[index_remain]
    return vor

def judgePassObstacle(pos1, pos2, obstacle_map):
    rr, cc = line(int(pos1[1]), int(pos1[0]), int(pos2[1]), int(pos2[0]))
    passed_occupied = np.any(obstacle_map[rr, cc] == 1)
    if passed_occupied:
        return False
    else :
        return True

def simplify_graph(G):
    # 创建一个副本以避免在迭代时修改图
    H = G.copy()
    # 查找所有度数为2的中间节点
    nodes_to_remove = [node for node in H.nodes() if H.degree(node) == 2]
    
    for node in nodes_to_remove:
       while H.degree(node) == 2:  # 确保处理后的节点度数仍然为2
            neighbors = list(H.neighbors(node))
            if len(neighbors) == 2:
                # 添加一条边连接中间节点的邻居
                H.add_edge(neighbors[0], neighbors[1])
                # 移除节点的所有边
                H.remove_edge(node, neighbors[0])
                H.remove_edge(node, neighbors[1])

    H=remove_isolated_nodes(H)
    return H

def simplify_graph2(G):
    # 创建一个副本以避免在迭代时修改图
    H = G.copy()

    pos = nx.get_node_attributes(H, 'pos')
    nodes_to_remove = [node for node in H.nodes() if H.degree(node) == 2]
    for node in nodes_to_remove:
       while H.degree(node) == 2:  # 确保处理后的节点度数仍然为2
            neighbors = list(H.neighbors(node))
            if len(neighbors) == 2:
                # 添加一条边连接中间节点的邻居
                vector1 = np.array(pos[neighbors[0]]) - np.array(pos[node])
                vector2 = np.array(pos[neighbors[1]]) - np.array(pos[node])
                # 计算夹角
                angle = calculate_angle_between_vectors(vector1, vector2)
                if angle > 150 :
                    H.add_edge(neighbors[0], neighbors[1])
                    # 移除节点的所有边
                    H.remove_edge(node, neighbors[0])
                    H.remove_edge(node, neighbors[1])
                if angle <= 150 :
                    break
    H=remove_isolated_nodes(H)

    return H


def extract_semantic_info_for_landmarks(landmark_nodes, objects_post, object_nodes,
                                         room_nodes, group_nodes, edges,
                                         search_radius=50.0, map_resolution=5.0):
    """Extract semantic information from scene graph for each landmark.

    Args:
        landmark_nodes: np.ndarray of shape (N, 2), landmark positions in map coordinates
        objects_post: MapObjectList containing all detected objects
        object_nodes: List of object nodes from scene graph
        room_nodes: List of room nodes
        group_nodes: List of group nodes
        edges: List of edges (relationships) in the scene graph
        search_radius: Distance threshold for considering objects/rooms as "nearby"

    Returns:
        List of dicts, one per landmark, each containing:
            - 'nearby_objects': list of (object_info, distance) tuples
            - 'nearby_rooms': list of (room_info, distance) tuples
            - 'nearby_edges': list of relevant edges (relationships)
            - 'object_count': number of nearby objects
            - 'dominant_room': most likely room type
    """
    if landmark_nodes is None or len(landmark_nodes) == 0:
        return []

    semantic_info_list = []

    for landmark_pos in landmark_nodes:
        landmark_info = {
            'nearby_objects': [],
            'nearby_rooms': [],
            'nearby_edges': [],
            'object_count': 0,
            'dominant_room': None
        }

        # Find nearby objects
        for obj_node in object_nodes:
            # Use node.center instead of bbox_center
            if not hasattr(obj_node, 'center') or obj_node.center is None:
                continue

            # IMPORTANT: obj_node.center is [x, y] = [col, row]
            # But landmark_pos is [row, col]
            # Need to swap to make them comparable
            obj_pos = np.array([obj_node.center[1], obj_node.center[0]])  # [y, x] = [row, col]
            distance = np.linalg.norm(landmark_pos - obj_pos)

            if distance <= search_radius:
                obj_info = {
                    'caption': obj_node.object.get('captions', ['unknown'])[0] if 'captions' in obj_node.object else 'unknown',
                    'captions': obj_node.object.get('captions', ['unknown']),
                    'position': obj_pos.tolist(),  # Now in [row, col] format
                    'position_original': obj_node.center,  # Keep original [x, y] for reference
                    'distance': float(distance),  # Distance in pixels
                    'distance_m': float(distance * map_resolution / 100.0)  # Distance in meters
                }
                landmark_info['nearby_objects'].append(obj_info)

        landmark_info['object_count'] = len(landmark_info['nearby_objects'])

        # Sort objects by distance
        landmark_info['nearby_objects'].sort(key=lambda x: x['distance'])

        # Find nearby rooms
        room_distance_map = {}
        for room_node in room_nodes:
            # Get room type from caption attribute
            room_type = getattr(room_node, 'caption', None) or getattr(room_node, 'room_type', None)
            if room_type is None:
                continue

            # Try to get room center if available
            if hasattr(room_node, 'center') and room_node.center is not None:
                room_center = np.array(room_node.center)
                distance = np.linalg.norm(landmark_pos - room_center)

                if distance <= search_radius * 2:  # Use larger radius for rooms
                    if room_type not in room_distance_map or distance < room_distance_map[room_type]['distance']:
                        room_distance_map[room_type] = {
                            'room_type': room_type,
                            'distance': float(distance),
                            'confidence': getattr(room_node, 'confidence', 0.5)
                        }
            else:
                # If room has no center, check if it contains nearby objects
                if hasattr(room_node, 'nodes') and len(room_node.nodes) > 0:
                    # Calculate average position of objects in this room
                    room_obj_positions = []
                    for obj_node in room_node.nodes:
                        if hasattr(obj_node, 'center') and obj_node.center is not None:
                            # Remember: obj_node.center is [x, y] = [col, row], need to swap
                            room_obj_positions.append([obj_node.center[1], obj_node.center[0]])

                    if room_obj_positions:
                        room_center = np.mean(room_obj_positions, axis=0)  # Now in [row, col] format
                        distance = np.linalg.norm(landmark_pos - room_center)

                        if distance <= search_radius * 2:
                            if room_type not in room_distance_map or distance < room_distance_map[room_type]['distance']:
                                room_distance_map[room_type] = {
                                    'room_type': room_type,
                                    'distance': float(distance),
                                    'confidence': 0.6
                                }

        landmark_info['nearby_rooms'] = list(room_distance_map.values())
        landmark_info['nearby_rooms'].sort(key=lambda x: x['distance'])

        # Determine dominant room
        if landmark_info['nearby_rooms']:
            landmark_info['dominant_room'] = landmark_info['nearby_rooms'][0]['room_type']

        # Find relevant edges (relationships involving nearby objects)
        nearby_obj_captions = [obj['caption'] for obj in landmark_info['nearby_objects']]
        for edge in edges:
            # Edge format may vary, adapt as needed
            if hasattr(edge, 'source') and hasattr(edge, 'target'):
                if (edge.source in nearby_obj_captions or
                    edge.target in nearby_obj_captions):
                    edge_info = {
                        'source': edge.source,
                        'relation': getattr(edge, 'relation', 'related_to'),
                        'target': edge.target
                    }
                    landmark_info['nearby_edges'].append(edge_info)

        semantic_info_list.append(landmark_info)

    return semantic_info_list
