# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt

# def visualize_hmm(transition_matrix, initial_distribution, edge_threshold=0.1, save_path=None):
#     # Create directed graph
#     G = nx.DiGraph()

#     # Add nodes and assign colors based on initial state distribution
#     node_colors = []
#     for i in range(len(initial_distribution)):
#         G.add_node(i)
#         node_colors.append(initial_distribution[i])

#     # Add unidirectional edges based on transition probabilities above threshold
#     for i in range(len(transition_matrix)):
#         for j in range(len(transition_matrix[i])):
#             if transition_matrix[i][j] >= edge_threshold and transition_matrix[j][i] < edge_threshold:
#                 G.add_edge(i, j, weight=transition_matrix[i][j])
#             elif transition_matrix[i][j] < edge_threshold and transition_matrix[j][i] >= edge_threshold:
#                 G.add_edge(j, i, weight=transition_matrix[j][i])

#     # Define layout
#     pos = nx.circular_layout(G)

#     # Draw nodes with colors based on initial state distribution
#     nx.draw(G, pos, node_color=node_colors, cmap=plt.cm.Reds, with_labels=True, node_size=1000)

#     # Draw edges with labels
#     edge_labels = {(i, j): round(weight, 2) for i, j, weight in G.edges(data='weight')}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

#     # Save or show plot
#     if save_path is not None:
#         plt.savefig(save_path,dpi=300)
#     else:
#         plt.show()

# # Example transition matrix and initial distribution
# transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
# initial_distribution = np.array([0.8, 0.2])

# # Set edge threshold
# edge_threshold = 0.2

# # Save figure to desired location
# save_path = "plots/hmm_graph.png"

# # Visualize HMM and save figure
# visualize_hmm(transition_matrix, initial_distribution, edge_threshold, save_path)

# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import gravis as gv

# def create_graph(transition_matrix, initial_distribution, edge_threshold):
#     num_states = len(transition_matrix)

#     # Create a directed graph
#     G = nx.DiGraph()

#     # Add nodes
#     for i in range(num_states):
#         G.add_node(i, label=str(i))

#     # Add edges above the threshold
#     for i in range(num_states):
#         for j in range(num_states):
#             if transition_matrix[i][j] >= edge_threshold:
#                 G.add_edge(i, j, weight=transition_matrix[i][j])

#     return G

# def plot_graph_with_gravis(G, output_file=None):
#     # Convert NetworkX graph to Gravis format
#     graph_dict = {
#         'graph': {
#             'metadata': {
#                 'arrow_size': 5,
#                 'background_color': 'white',
#                 'edge_size': 3,
#                 'edge_label_size': 20,
#                 'edge_label_color': 'black',
#                 'node_size': 40,
#                 'node_color': 'red',
#                 'random_seed':0
#             },
#             'nodes': {str(node): {} for node in G.nodes()},
#             'edges': [{'source': str(u), 'target': str(v)} for u, v in G.edges()]
#         }
#     }

#     # Use Gravis to visualize the graph
#     fig = gv.vis(graph_dict,edge_curvature=0.4)

#     # Save or display the figure
#     if output_file:
#         # fig.export_html(output_file+'.html',overwrite=True)
#         fig.export_png(output_file+'.png',overwrite=True)
#         # fig.export_html(output_file,overwrite=True)
#         fig.export_svg(output_file+'.svg',overwrite=True)
#     else:
#         fig.display()

# def plot_graph(G, initial_distribution, save_path=None):
#     # Spring layout for better visualization
#     pos = nx.spring_layout(G,seed=0)

#     # Node colors based on initial state distribution
#     node_colors = [initial_distribution[node] for node in G.nodes()]

#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, cmap=plt.get_cmap('summer'))

#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_color='red')

#     # Draw edges
#     nx.draw_networkx_edges(G, pos, arrows=True, connectionstyle="arc3,rad=0.2")

#     # Add transition probabilities as annotations
#     edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)

#     # Save or display the figure
#     if save_path:
#         plt.savefig(save_path,dpi=300)
#     else:
#         plt.show()

# # Example transition matrix and initial distribution
# transition_matrix = np.array([[0.6, 0.4, 0.0],
#                                [0.3, 0.5, 0.2],
#                                [0.1, 0.3, 0.6]])

# initial_distribution = {0: 0.3, 1: 0.4, 2: 0.3}

# # Threshold value for edges
# edge_threshold = 0.2

# # Create graph with edges above threshold
# G = create_graph(transition_matrix, initial_distribution, edge_threshold)

# # Save the figure to a desired location
# save_path = "/home/kabird/hmm_analysis/plots/hmm_graph"
# plot_graph(G, initial_distribution, save_path=save_path+".pdf")
# # plot_graph_with_gravis(G, output_file=save_path)

save_path = "/home/kabird/hmm_analysis/plots/hmm_graph"

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Example Transition Matrix and Initial State Distribution
transition_matrix = np.array([
    [0.1, 0.6, 0.3],
    [0.4, 0.1, 0.5],
    [0.3, 0.3, 0.4]
])
initial_distribution = np.array([0.2, 0.5, 0.3])

# Threshold for edge removal
edge_threshold = 0.3


def create_graph(transition_matrix, initial_distribution, edge_threshold, save_path=None):
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
    color_map = [plt.cm.viridis(G.nodes[i]['color']) for i in G.nodes]

    # Draw the graph
    # pos = nx.circular_layout(G)  # positions for all nodes
    pos = nx.spring_layout(G, seed=0)
    edges = G.edges(data=True)

    # Node size scaled by initial_distribution
    node_sizes = [value * 600 for value in initial_distribution]
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=700)  # 700)

    # nx.draw_networkx_labels(G, pos, font_color='red')

    # Edge width scaled by weight
    edge_widths = [d['weight'] * 6 + 1 for (_, _, d) in edges]  # Scale factor of 5 for visibility

    # Draw curved edges for bidirectional and straight for unidirectional
    for (u, v, d) in edges:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_widths.pop(0),
                               connectionstyle=d['connectionstyle'], arrowstyle='-|>',
                               arrowsize=20, edge_color='gray')

    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # edges = [f'{e:.2f}' for e in edge_labels]
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # plt.title('Graph Visualization of HMM')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)


def create_graph2(transition_matrix, initial_distribution, edge_threshold, save_path=None, seed=0):
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
    pos = nx.spring_layout(G, seed=seed)
    edges = G.edges(data=True)
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=1400, zorder=1)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=25, zorder=2)

    # Edge width scaled by weight
    edge_widths = [d['weight'] * 6 + 1 for (_, _, d) in edges]  # Scale factor of 5 for visibility

    # Draw curved edges for bidirectional and straight for unidirectional
    for (u, v, d) in edges:
        alpha_value = min(1.0, d['weight'] / 0.5)  # Scale alpha based on weight
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_widths.pop(0),
                               connectionstyle=d['connectionstyle'], arrowstyle='-|>',
                               arrowsize=20, edge_color='gray', alpha=alpha_value, zorder=3)

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, transparent=True, dpi=300)


import matplotlib.pyplot as plt
import networkx as nx


def create_graph2(transition_matrix, initial_distribution, edge_threshold, save_path=None, seed=0):
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

    # Draw the graph layout
    pos = nx.spring_layout(G, seed=seed)
    edges = G.edges(data=True)

    # Manually set zorder using matplotlib's scatter for nodes
    ax = plt.gca()  # Get current axis

    node_collection = nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=1400)
    node_collection.set_zorder(2)  # Set zorder manually for nodes

    # Draw labels and set their zorder manually
    label_dict = nx.draw_networkx_labels(G, pos, font_color='white', font_size=30)
    for label in label_dict.values():
        label.set_zorder(3)  # Manually set the zorder for each label

    # Edge width scaled by weight
    edge_widths = [d['weight'] * 6 + 1 for (_, _, d) in edges]

    # Draw edges manually using LineCollection from matplotlib for z-order control
    for (u, v, d) in edges:
        alpha_value = min(1.0, d['weight'] / 0.5)  # Scale alpha based on weight
        connectionstyle = d.get('connectionstyle', 'arc3,rad=0')
        # Use matplotlib to manually plot the edges
        line = nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], width=edge_widths.pop(0),
            connectionstyle=connectionstyle, arrowstyle='-|>', arrowsize=50,
            edge_color='gray', alpha=alpha_value
        )
        # You can access the created collection and adjust zorder if needed
        if isinstance(line, list):
            for ln in line:
                ln.set_zorder(1)  # Set zorder for edges

    # Add edge labels (optional)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    plt.axis('off')

    # plt.tight_layout()

    # Save the graph if save_path is provided
    if save_path:
        plt.savefig(save_path, transparent=True, dpi=300)

    plt.close()

# def create_graph2(transition_matrix, initial_distribution, edge_threshold, save_path=None):
#     # Create a directed graph
#     G = nx.DiGraph()

#     # Add nodes with color attribute
#     for i in range(len(initial_distribution)):
#         G.add_node(i, color=initial_distribution[i])

#     # Add edges filtering by edge_threshold
#     for i in range(transition_matrix.shape[0]):
#         for j in range(transition_matrix.shape[1]):
#             if transition_matrix[i, j] > edge_threshold:
#                 if G.has_edge(j, i):
#                     # Add curved edges if bidirectional
#                     G.add_edge(i, j, weight=transition_matrix[i, j], connectionstyle='arc3,rad=0.2')
#                 else:
#                     # Straight edge if unidirectional
#                     G.add_edge(i, j, weight=transition_matrix[i, j], connectionstyle='arc3,rad=0')

#     # Define color map based on initial distribution
#     color_map = [plt.cm.viridis(G.nodes[i]['color']) for i in G.nodes]

#     # Draw the graph
#     pos = nx.spring_layout(G)
#     edges = G.edges(data=True)

#     # Node size scaled by initial_distribution
#     # node_sizes = [value * 600 for value in initial_distribution]

#     # Draw nodes with size based on initial_distribution
#     nx.draw_networkx_nodes(G, pos, node_color=color_map)#, node_size=node_sizes)

#     # Draw labels with annotations below the node points
#     nx.draw_networkx_labels(G, pos, labels={node: f'{initial_distribution[node]:.2f}' for node in G.nodes()}, verticalalignment='top')

#     # Edge width scaled by weight
#     edge_widths = [d['weight'] * 6 + 1 for (_, _, d) in edges]  # Scale factor of 5 for visibility

#     # Draw curved edges for bidirectional and straight for unidirectional
#     for (u, v, d) in edges:
#         alpha_value = min(1.0, d['weight'] / 2.0)  # Scale alpha based on weight
#         nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_widths.pop(0),
#                                connectionstyle=d['connectionstyle'], arrowstyle='-|>',
#                                arrowsize=20, edge_color='gray', alpha=alpha_value)

#     edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
#     plt.axis('off')
#     if save_path:
#         plt.savefig(save_path,dpi=300)

# Call the function
# create_graph(transition_matrix, initial_distribution, edge_threshold)