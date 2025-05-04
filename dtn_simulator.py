import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def main():
    st.title("Optical Network Topology Visualizer")
    
    # Sidebar controls
    st.sidebar.header("Network Configuration")
    topology_type = st.sidebar.selectbox(
        "Select Topology Type",
        ["Star", "Ring", "Mesh"],
        index=0
    )
    
    num_nodes = st.sidebar.slider(
        "Number of Nodes",
        min_value=3,
        max_value=15,
        value=5,
        step=1
    )
    
    # Simulation parameters
    st.sidebar.header("Simulation Parameters")
    traffic_intensity = st.sidebar.slider(
        "Traffic Intensity",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )
    
    protection_switch = st.sidebar.checkbox(
        "Enable Protection Switching",
        value=True
    )
    
    # Create the network graph
    G = create_topology(topology_type, num_nodes)
    
    # Visualize the topology
    st.header("Network Topology Visualization")
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color='gray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)
    
    # Add edge weights (latency)
    edge_labels = nx.get_edge_attributes(G, 'latency')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    
    st.pyplot(fig)
    
    # Network metrics
    st.header("Network Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Nodes", num_nodes)
    
    with col2:
        avg_latency = calculate_average_latency(G)
        st.metric("Average Latency (ms)", f"{avg_latency:.2f}")
    
    with col3:
        total_bandwidth = calculate_total_bandwidth(G)
        st.metric("Total Bandwidth (Gbps)", f"{total_bandwidth:.2f}")
    
    # Traffic simulation
    st.header("Traffic Simulation")
    if st.button("Run Traffic Simulation"):
        simulate_traffic(G, traffic_intensity)
    
    # Protection switching simulation
    if protection_switch:
        st.header("Protection Switching Simulation")
        if st.button("Simulate Link Failure"):
            simulate_link_failure(G, pos)

def create_topology(topology_type, num_nodes):
    """Create network topology based on selected type"""
    G = nx.Graph()
    
    # Add nodes
    nodes = [f"Node {i}" for i in range(1, num_nodes+1)]
    G.add_nodes_from(nodes)
    
    # Add edges based on topology type
    if topology_type == "Star":
        # Connect all nodes to the central node (Node 1)
        center = nodes[0]
        for node in nodes[1:]:
            latency = np.random.randint(1, 10)
            bandwidth = np.random.randint(10, 100)
            G.add_edge(center, node, latency=latency, bandwidth=bandwidth)
    
    elif topology_type == "Ring":
        # Connect nodes in a ring
        for i in range(num_nodes):
            node1 = nodes[i]
            node2 = nodes[(i+1)%num_nodes]
            latency = np.random.randint(1, 10)
            bandwidth = np.random.randint(10, 100)
            G.add_edge(node1, node2, latency=latency, bandwidth=bandwidth)
    
    elif topology_type == "Mesh":
        # Connect all nodes to each other
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                latency = np.random.randint(1, 10)
                bandwidth = np.random.randint(10, 100)
                G.add_edge(nodes[i], nodes[j], latency=latency, bandwidth=bandwidth)
    
    return G

def calculate_average_latency(G):
    """Calculate average latency across all edges"""
    latencies = [d['latency'] for u, v, d in G.edges(data=True)]
    return np.mean(latencies) if latencies else 0

def calculate_total_bandwidth(G):
    """Calculate total bandwidth across all edges"""
    bandwidths = [d['bandwidth'] for u, v, d in G.edges(data=True)]
    return np.sum(bandwidths) if bandwidths else 0

def simulate_traffic(G, intensity):
    """Simulate traffic routing and display results"""
    # Select random source and destination
    nodes = list(G.nodes())
    src = np.random.choice(nodes)
    dst = np.random.choice([n for n in nodes if n != src])
    
    # Find shortest path based on latency
    try:
        path = nx.shortest_path(G, source=src, target=dst, weight='latency')
        path_edges = list(zip(path[:-1], path[1:]))
        
        # Calculate total latency and bandwidth
        total_latency = sum(G[u][v]['latency'] for u, v in path_edges)
        min_bandwidth = min(G[u][v]['bandwidth'] for u, v in path_edges)
        
        # Display results
        st.success(f"Traffic routed from {src} to {dst}")
        st.write(f"Path: {' → '.join(path)}")
        st.write(f"Total Latency: {total_latency} ms")
        st.write(f"Bottleneck Bandwidth: {min_bandwidth} Gbps")
        
        # Visualize the path
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, ax=ax, width=1, edge_color='gray')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)
        
        # Highlight the path
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='red', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[src, dst], node_size=1000, node_color=['green', 'blue'], ax=ax)
        
        st.pyplot(fig)
        
    except nx.NetworkXNoPath:
        st.error(f"No path exists between {src} and {dst}")

def simulate_link_failure(G, pos):
    """Simulate link failure and protection switching"""
    if len(G.edges()) == 0:
        st.warning("No edges to fail in this topology")
        return
    
    # Select a random edge to fail
    edges = list(G.edges())
    failed_edge = edges[np.random.randint(0, len(edges))]
    u, v = failed_edge
    
    # Store original attributes
    original_latency = G[u][v]['latency']
    original_bandwidth = G[u][v]['bandwidth']
    
    # Fail the edge by setting high latency and zero bandwidth
    G[u][v]['latency'] = 999
    G[u][v]['bandwidth'] = 0
    
    # Find alternative paths
    nodes = list(G.nodes())
    paths_affected = []
    
    for src in nodes:
        for dst in nodes:
            if src != dst:
                try:
                    path = nx.shortest_path(G, source=src, target=dst, weight='latency')
                    if (u in path and v in path) or (v in path and u in path):
                        paths_affected.append((src, dst, path))
                except nx.NetworkXNoPath:
                    continue
    
    # Visualize the failure and protection switching
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, ax=ax, width=1, edge_color='gray')
    
    # Highlight the failed edge
    nx.draw_networkx_edges(G, pos, edgelist=[failed_edge], width=3, edge_color='black', style='dashed', ax=ax)
    
    # Highlight affected paths
    for src, dst, path in paths_affected[:3]:  # Show max 3 affected paths
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2, edge_color='orange', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[src, dst], node_size=1000, node_color=['green', 'blue'], ax=ax)
    
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)
    
    st.warning(f"Link between {u} and {v} has failed!")
    st.pyplot(fig)
    
    # Show protection switching information
    if paths_affected:
        st.subheader("Protection Switching Activated")
        st.write(f"Failed Link: {u} ↔ {v}")
        st.write(f"Affected Paths: {len(paths_affected)}")
        
        # Show some rerouted paths
        st.write("Example rerouted paths:")
        for src, dst, path in paths_affected[:3]:
            st.write(f"{src} → {dst}: {' → '.join(path)}")
    else:
        st.info("No active paths affected by this link failure")
    
    # Restore the original edge attributes
    G[u][v]['latency'] = original_latency
    G[u][v]['bandwidth'] = original_bandwidth

if __name__ == "main":
    main()