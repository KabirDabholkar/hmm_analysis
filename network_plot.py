import networkx as nx
import matplotlib.pyplot as plt

def create_graph2(transition_matrix, initial_distribution, edge_threshold, save_path=None):
    print('trans mat:', transition_matrix.shape, 'init dist:', initial_distribution.shape)
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with color attribute
    for i in range(len(initial_distribution)):
        G.add_node(i, color=initial_distribution[i])

    # Add edges filtering by edge_threshold
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            if transition_matrix[i, j] > edge_threshold:
                if G.has_edge(j, i):
                    # Add curved edges if bidirectional
                    G.add_edge(i, j, weight=transition_matrix[i, j], connectionstyle='arc3,rad=0.2')
                else:
                    # Straight edge if unidirectional
                    G.add_edge(i, j, weight=transition_matrix[i, j], connectionstyle='arc3,rad=0')

    # Define color map based on initial distribution
    color_map = [plt.cm.hot(G.nodes[i]['color']) for i in G.nodes]

    # Draw the graph
    # pos = nx.circular_layout(G)  # positions for all nodes
    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=1400)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=25)

    # Edge width scaled by weight
    edge_widths = [d['weight'] * 6 + 1 for (_, _, d) in edges]  # Scale factor of 5 for visibility

    # Draw curved edges for bidirectional and straight for unidirectional
    for (u, v, d) in edges:
        alpha_value = min(1.0, d['weight'] / 0.5)  # Scale alpha based on weight
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_widths.pop(0),
                               connectionstyle=d['connectionstyle'], arrowstyle='-|>',
                               arrowsize=20, edge_color='gray', alpha=alpha_value)

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, transparent=True, dpi=300)