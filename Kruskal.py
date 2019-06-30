# Código retirado de: https://gist.github.com/hayderimran7/09960ca438a65a9bd10d0254b792f48f

parent = dict()
rank = dict()

def make_set(vertice):
    parent[vertice] = vertice
    rank[vertice] = 0

def find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice])
    return parent[vertice]

def union(vertice1, vertice2):
	root1 = find(vertice1)
	root2 = find(vertice2)
	if root1 != root2:
		if rank[root1] > rank[root2]:
			parent[root2] = root1

	else:
		parent[root1] = root2

	if rank[root1] == rank[root2]:
		rank[root2] += 1

def kruskal(graph):
	for vertice in graph['vertices']:
		make_set(vertice)
		minimum_spanning_tree = set()
		edges = list(graph['edges'])
		edges.sort()

	for edge in edges:
		weight, vertice1, vertice2 = edge
		if find(vertice1) != find(vertice2):
			union(vertice1, vertice2)
			minimum_spanning_tree.add(edge)

	return sorted(minimum_spanning_tree)		

def print_kruskal(graph):
	k = kruskal(graph)
	print('Resultado do algoritmo de kruskal:')

	for node in k:
		print(node)

def print_graph(graph):
	print('Grafo original:')
	
	vert = []
	for v in graph['vertices']:
		vert.append(v)
	print('Vertices: {}'.format(vert))

	print('Edges: ')
	for e in graph['edges']:
		print(e)


graph = {
	'vertices': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
	'edges': set([
	(7, 'A', 'B'),
	(5, 'A', 'D'),
	(7, 'B', 'A'),
	(8, 'B', 'C'),
	(9, 'B', 'D'),
	(7, 'B', 'E'),
	(8, 'C', 'B'),
	(5, 'C', 'E'),
	(5, 'D', 'A'),
	(9, 'D', 'B'),
	(7, 'D', 'E'),
	(6, 'D', 'F'),
	(7, 'E', 'B'),
	(5, 'E', 'C'),
	(15, 'E', 'D'),
	(8, 'E', 'F'),
	(9, 'E', 'G'),
	(6, 'F', 'D'),
	(8, 'F', 'E'),
	(11, 'F', 'G'),
	(9, 'G', 'E'),
	(11, 'G', 'F'),
	])
}

print_graph(graph)
print_kruskal(graph)