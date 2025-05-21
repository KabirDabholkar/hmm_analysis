import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def state_transition_counter(state_sequence_array, M):
    # Initialize transition matrix and state visit count vector
    transition_matrix = np.zeros((M, M), dtype=int)
    state_visit_count = np.zeros(M, dtype=int)

    # Loop through each trial sequence
    for trial in state_sequence_array:
        # Flatten the array to ensure it is a 1D sequence of states
        states = trial.flatten()

        # Update the state visit counts
        for state in states:
            state_visit_count[state] += 1

        # Update the transition matrix for each consecutive state pair
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_matrix[current_state, next_state] += 1

    return transition_matrix, state_visit_count


def plot_state_transition_graph_old(transition_matrix, state_visit_count, savepath='state_transition_graph'):
    # Create a directed graph
    G = nx.DiGraph()

    # Number of states
    M = len(state_visit_count)

    # Add nodes with sizes proportional to state visit count
    for state in range(M):
        G.add_node(state, size=state_visit_count[state])

    # Add edges with weight proportional to transition counts
    max_transition = np.max(transition_matrix) if np.max(transition_matrix) > 0 else 1
    for from_state in range(M):
        for to_state in range(M):
            if transition_matrix[from_state, to_state] > 0:
                # Normalize width
                width = 5 * (transition_matrix[from_state, to_state] / max_transition)
                G.add_edge(from_state, to_state, weight=width)

    # Define the node sizes and edge widths
    node_sizes = [state_visit_count[state] * 100 for state in range(M)]
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]

    # Plot the graph using a circular layout
    pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)

    # Draw edges with explicit color and width to avoid artifacts
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='->',
        arrowsize=10,
        width=edge_widths,
        edge_color='gray',  # Set a fixed color to avoid unwanted fills
        connectionstyle='arc3,rad=0.2',
    )

    # Remove axis and margins
    plt.gca().set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the figure as PNG and PDF
    plt.savefig(f"{savepath}.png", format='png')
    plt.savefig(f"{savepath}.pdf", format='pdf')

    # Close the plot to free up resources
    plt.close()


# Example usage:

def plot_state_transition_graph(transition_matrix, initial_distribution, edge_threshold=0.05, savepath=None, seed=0):
    print('trans mat:', transition_matrix.shape, 'init dist:', initial_distribution.shape)
    # Create a directed graph
    plt.figure(figsize=(7, 6))
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

    node_sizes = 1e6 * initial_distribution
    node_collection = nx.draw_networkx_nodes(G, pos,
                                             node_size=np.sqrt(node_sizes))  # node_size=1400 #node_color=color_map,
    # node_collection.set_zorder(2)  # Set zorder manually for nodes

    # Draw labels and set their zorder manually
    # label_dict = nx.draw_networkx_labels(G, pos, font_color='white', font_size=30)
    # for label in label_dict.values():
    #     label.set_zorder(3)  # Manually set the zorder for each label

    # Edge width scaled by weight
    edge_widths = [d['weight'] * 100 for (_, _, d) in edges]

    # Draw edges manually using LineCollection from matplotlib for z-order control
    for (u, v, d) in edges:
        alpha_value = min(1.0, d['weight'] * 10)  # Scale alpha based on weight

        connectionstyle = d.get('connectionstyle', 'arc3,rad=0')
        # Use matplotlib to manually plot the edges
        line = nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], width=3,  # width=edge_widths.pop(0),
            connectionstyle=connectionstyle, arrowstyle='-|>', arrowsize=20,
            edge_color='gray', alpha=alpha_value
        )
        # You can access the created collection and adjust zorder if needed
        # if isinstance(line, list):
        #     for ln in line:
        #         ln.set_zorder(1)  # Set zorder for edges

    # Add edge labels (optional)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    plt.axis('off')

    # plt.tight_layout()

    # Save the graph if savepath is provided
    if savepath:
        plt.savefig(savepath, transparent=True, dpi=300)

    plt.close()


def plot_state_transition_graph(transition_matrix, initial_distribution, edge_threshold=0.05, savepath=None,
                                spring_kwargs={}):
    print('trans mat:', transition_matrix.shape, 'init dist:', initial_distribution.shape)

    # Create a directed graph
    plt.figure(figsize=(7, 6))
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
    pos = nx.spring_layout(G, **spring_kwargs)
    edges = G.edges(data=True)

    # Manually set zorder using matplotlib's scatter for nodes
    ax = plt.gca()  # Get current axis

    node_sizes = 1e6 * initial_distribution

    # Edge width scaled by weight
    edge_weights = [d['weight'] for (_, _, d) in edges]

    # Normalize edge weights for color mapping (from 0 to 1)
    # max_weight = max(edge_weights)
    max_weight = 0.25
    # min_weight = min(edge_weights)
    min_weight = 0  # min(edge_weights)
    norm_edge_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
    norm_node_values = [(w - min_weight) / (max_weight - min_weight) for w in initial_distribution]

    # Create a colormap from white (low values) to black (high values)
    # edge_colors = [plt.cm.gray(1 - norm_w) for norm_w in norm_edge_weights]
    # edge_colors = [plt.cm.gray(0.2 + 0.8 * (1 - norm_w)) for norm_w in norm_edge_weights]
    edge_colors = [plt.cm.inferno_r(0.2 + 0.8 * (1 - norm_w)) for norm_w in norm_edge_weights]
    node_colors = [plt.cm.inferno_r(0.2 + 0.8 * (1 - norm_w)) for norm_w in norm_node_values]
    # edge_colors = [plt.cm.inferno(norm_w) for norm_w in norm_edge_weights]

    # Draw nodes with specified colours and sizes
    node_collection = nx.draw_networkx_nodes(G, pos, node_size=np.sqrt(node_sizes), node_color=node_colors)

    # Draw edges with the color based on normalized weight
    for (u, v, d), color in zip(edges, edge_colors):
        alpha_value = min(1.0, d['weight'] * 10)
        connectionstyle = d.get('connectionstyle', 'arc3,rad=0')
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            # width=3,
            width=d['weight'] * 20,
            connectionstyle=connectionstyle,
            arrowstyle='-|>', arrowsize=50,  # d['weight']*200,
            edge_color=[color],
            # edge_color='grey',
            alpha=1  # alpha_value  # Use color from the grayscale colormap
        )

    plt.axis('off')

    # Save the graph if savepath is provided
    if savepath:
        plt.savefig(savepath, transparent=True, dpi=300)

    plt.show()
    plt.close()

