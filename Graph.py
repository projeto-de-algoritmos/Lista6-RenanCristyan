# Código retirado de: https://gist.github.com/cauemello/208625ce41b61655d4db76f54f1d7fe5#file-graph-py 

import math
from queue import PriorityQueue
from random import randint
import networkx as nx
import statistics
import matplotlib.pyplot as plt

# Implementacao de Grafo baseada em http://www.python-course.eu/graphs_python.php
class Graph(object):

    def __init__(self, graph_dict={}):
        self.__graph_dict = graph_dict

    def vertices(self):
        return list(self.__graph_dict.keys())

    def show_vertices(self):
        print('All vertices:')
        all_vertices = []
        for vertice in self.vertices():
            all_vertices.append(vertice)
        print(sorted(all_vertices))

    def edges(self):
        return self.__generate_edges()

    def show_edges(self):
        print('All edges:')
        edges = self.__generate_edges()
        for edge in edges:
            print(edge)

    def add_vertex(self, vertex):
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []

    def add_edge(self, *edge, bidirectional=True):
        (vertex1, vertex2, cost) = edge
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)
        self.__add_edge_no_repetition(vertex1, vertex2, cost)
        if bidirectional:
            self.__add_edge_no_repetition(vertex2, vertex1, cost)

    def direct_cost(self, vertex1, vertex2):
        list_v1 = self.__graph_dict[vertex1]
        for (v, cost) in list_v1:
            if v == vertex2:
                return cost
        else:
            return math.inf

    def __add_edge_no_repetition(self, v1, v2, cost):
        list_v1 = self.__graph_dict[v1]
        for i, (v, _) in enumerate(list_v1):
            if v == v2:
                list_v1[i] = (v2, cost)
                break
        else:
            list_v1.append((v2, cost))

    def __generate_edges(self):
        edges = []
        for vertex in self.__graph_dict:
            for (neighbour, cost) in self.__graph_dict[vertex]:
                if (neighbour, vertex) not in edges:
                    edges.append((vertex, neighbour, cost))
        return edges

    def __str__(self):
        return 'Vertices: {0}\nEdges: {1}'.format(sorted(self.vertices()), sorted(self.edges()))

def dijkstra(graph, root):

    queue = PriorityQueue()  # Lista de prioridades

    path = {}  # Dicionário com o caminho e o custo total
    for v in graph.vertices():
        if v == root:
            path[v] = [[], 0]  # Custo 0 para o root
        else:
            path[v] = [[], math.inf]  # Custo infinito para os demais

        queue.put((path[v][1], v))  # Adiciona todas na lista de prioridade (maior prioridade = menor custo)

    remaing_vertices = list(graph.vertices())  # lista de vertices nao visitados

    for i in range(len(graph.vertices())):
        u = queue.get()[1]  # vertice prioritario da lista
        remaing_vertices.remove(u)  # remove da lista de nao visitados

        for v in remaing_vertices:  # para cada v nao visitado
            du = path[u][1]  # menor custo ate vertice u (prioritario)
            w = graph.direct_cost(u, v)  # custo de u ate v
            dv = path[v][1]  # menor custo ate vertice v
            if du + w < dv:  # O caminho até v pelo u é menos custoso que o melhor até então?
                path[v][1] = du + w  # Atualiza o custo
                path[v][0] = path[u][0] + [u]  # Atualiza o caminho
                queue.queue.remove((dv, v))  # Atualiza a prioridade do vertice v na lista de prioridade
                queue.put((path[v][1], v))

    return path

def print_dijkstra(graph):
    for node in g.vertices():
        print('Caminhos mais curtos desde o vertice {}:\n{}'.format(node, path_as_string(dijkstra(g, node))))
        print('--')

def path_as_string(path):
    path_tidy = []
    vertices = sorted(path.keys())
    for v in vertices:
        cost = path[v][1]
        if cost == 0:
            continue
        p = '-'.join(path[v][0]) + '-' + v
        path_tidy.append(p + ' custo: ' + str(cost))
    return '\n'.join(path_tidy)

def prim(graph, root):
    vertex = [root]  # Lista dos vertices a partir do qual buscamos as arestas
    selected_edges = []  # Lista com as arestas selecionadas

    weight = 0  # Peso do minimum spanning tree

    remaing_vertices = list(graph.vertices())  # Lista com os vertices destinos da busca
    remaing_vertices.remove(root)  # O root eh ponto de partida, entao sai da lista

    for i in range(len(remaing_vertices)):  # Devemos buscar |V| - 1 vertices
        min_cost = math.inf  # Inicializamos o custo minimo como infinito
        va, vb = None, None  # Vertices candidatos para a aresta selecionada
        for v1 in vertex:  # Para cada vertice na lista de busca origem
            for v2 in remaing_vertices:  # Buscamos os vertices que ainda nao estao no grafo final
                cost = graph.direct_cost(v1, v2)  # Calcula o custo da aresta
                if cost < min_cost:  # Se for menor que o minimo ate entao, atualizamos os dados
                    va = v1
                    vb = v2
                    min_cost = cost

        if min_cost < math.inf:  # Depois de todas as buscas, se o custo eh finito:
            selected_edges.append((va, vb, min_cost))  # Adicionamos a aresta de va a vb na solucao
            vertex.append(vb)  # vb agora sera nova origem de busca
            remaing_vertices.remove(vb)  # vb nao mais sera destino de busca, pois ja consta na solucao
            weight += min_cost  # Atualiza o peso

    return selected_edges, weight  # Retorna a lista de arestas selecionadas com o peso total

def generate_random_graph(nodes, max_cost=20):

    l = len(nodes)
    _g = Graph({})

    for i in range(l):
        r1 = randint(1, l - 1)
        r2 = randint(1, l - 1)

        n1 = nodes[i]
        n2 = nodes[(i + r1) % l]
        n3 = nodes[(i + r2) % l]

        _g.add_edge(n1, n2, randint(1, max_cost))
        _g.add_edge(n1, n3, randint(1, max_cost))

    return _g

# Implementacao baseada no exemplo disponivel em:
# https://networkx.readthedocs.io/en/stable/examples/drawing/weighted_graph.html
def save_graph_as_png(graph, name=False):
    print('\n')
    nxg = nx.Graph()
    costs = []
    for (a, b, cost) in graph.edges():
        nxg.add_edge(a, b, cost=cost)
        costs.append(cost)

    pos = nx.spring_layout(nxg)  # positions for all nodes

    avg_cost = statistics.mean(costs)

    elarge = [(u, v) for (u, v, d) in nxg.edges(data=True) if d['cost'] > avg_cost]
    esmall = [(u, v) for (u, v, d) in nxg.edges(data=True) if d['cost'] <= avg_cost]

    # nodes
    nx.draw_networkx_nodes(nxg, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(nxg, pos, edgelist=elarge, width=4)
    nx.draw_networkx_edges(nxg, pos, edgelist=esmall, width=4, alpha=0.5, edge_color='b', style='dashed')

    # labels
    nx.draw_networkx_labels(nxg, pos, font_size=20, font_family='sans-serif')
    nx.draw_networkx_edge_labels(nxg, pos)

    plt.axis('off')
    
    if name:
        plt.savefig("{}.png".format(name))
    else:
        plt.savefig("graph.png")  # save as png

if __name__ == '__main__':

    g = Graph({})
    edges = [('a', 'b', 17), ('a', 'e', 14), ('a', 'h', 5), ('b', 'g', 18), ('b', 'h', 13), ('c', 'e', 20),
             ('c', 'f', 2), ('d', 'e', 19), ('d', 'g', 8), ('e', 'g', 12), ('f', 'g', 1), ('f', 'h', 13)]
    for e in edges:
        g.add_edge(*e)

    g_prim = Graph({})
    prim, w = prim(g, 'a')  # Retorna as arestas e o peso
    for e in prim:
        g_prim.add_edge(*e)

    print('Grafo Original:')
    g.show_vertices()
    g.show_edges()
    print('--')

    print_dijkstra(g)
    print('--')
    
    print('Minimal Spanning Tree (Peso Final = %s):' % w)
    g_prim.show_vertices()
    g_prim.show_edges()

    save_graph_as_png(g_prim, 'arvore_geradora_minima')