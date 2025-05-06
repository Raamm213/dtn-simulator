import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from collections import defaultdict

def main():
    st.title("Delay Tolerant Network (DTN) with Modulation and Noise Effects")
    
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

    # Modulation & Noise Controls
    st.sidebar.header("Modulation and Noise")
    modulation = st.sidebar.selectbox(
        "Modulation Scheme",
        ["BPSK", "QPSK", "16-QAM"],
        index=0
    )
    snr_db = st.sidebar.slider("SNR (dB)", min_value=0, max_value=30, value=10)
    pre_emphasis = st.sidebar.checkbox("Enable Pre-emphasis/De-emphasis", value=False)

    if pre_emphasis:
        snr_db += 3

    ber = calculate_ber(modulation, snr_db)
    
    # Create topology
    G = create_topology(topology_type, num_nodes)
    
    # Visualize topology
    st.header("Network Topology Visualization")
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color='gray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)
    edge_labels = nx.get_edge_attributes(G, 'latency')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    st.pyplot(fig)
    
    # Network metrics
    st.header("Network Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Nodes", num_nodes)
    with col2:
        avg_latency = calculate_average_latency(G)
        st.metric("Average Latency (ms)", f"{avg_latency:.2f}")
    with col3:
        total_bandwidth = calculate_total_bandwidth(G)
        st.metric("Total Bandwidth (Gbps)", f"{total_bandwidth:.2f}")
    with col4:
        st.metric("BER", f"{ber:.5f}")
    
    # Traffic simulation
    st.header("Traffic Simulation")
    if st.button("Run Traffic Simulation"):
        simulate_traffic(G, traffic_intensity, modulation, snr_db)
        modulation_visualization(modulation, snr_db)
    
    if protection_switch:
        st.header("Protection Switching Simulation")
        if st.button("Simulate Link Failure"):
            simulate_link_failure(G, pos)

def create_topology(topology_type, num_nodes):
    G = nx.Graph()
    nodes = [f"Node {i}" for i in range(1, num_nodes+1)]
    G.add_nodes_from(nodes)

    if topology_type == "Star":
        center = nodes[0]
        for node in nodes[1:]:
            latency = np.random.randint(1, 10)
            bandwidth = np.random.randint(10, 100)
            G.add_edge(center, node, latency=latency, bandwidth=bandwidth)
    elif topology_type == "Ring":
        for i in range(num_nodes):
            node1 = nodes[i]
            node2 = nodes[(i+1)%num_nodes]
            latency = np.random.randint(1, 10)
            bandwidth = np.random.randint(10, 100)
            G.add_edge(node1, node2, latency=latency, bandwidth=bandwidth)
    elif topology_type == "Mesh":
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                latency = np.random.randint(1, 10)
                bandwidth = np.random.randint(10, 100)
                G.add_edge(nodes[i], nodes[j], latency=latency, bandwidth=bandwidth)
    return G

def calculate_average_latency(G):
    latencies = [d['latency'] for u, v, d in G.edges(data=True)]
    return np.mean(latencies) if latencies else 0

def calculate_total_bandwidth(G):
    bandwidths = [d['bandwidth'] for u, v, d in G.edges(data=True)]
    return np.sum(bandwidths) if bandwidths else 0

def calculate_ber(modulation, snr_db):
    snr = 10 ** (snr_db / 10)
    if modulation == "BPSK":
        return 0.5 * np.exp(-snr)
    elif modulation == "QPSK":
        return 0.5 * np.exp(-snr / 2)
    elif modulation == "16-QAM":
        return (3/8) * np.exp(-snr / 10)
    return 0

def visualize_signal_waveforms(modulation, snr_db):
    st.subheader("Signal Waveform Visualization")
    
    # Generate a random bitstream
    num_bits = 100
    bits = np.random.randint(0, 2, num_bits)
    
    t = np.linspace(0, 1, num_bits*10)
    
    if modulation == "BPSK":
        signal = 2*bits - 1  # BPSK: 0->-1, 1->+1
        signal_upsampled = np.repeat(signal, 10)
        
        # Add noise
        snr_linear = 10 ** (snr_db / 10)
        noise_power = 1 / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal_upsampled))
        received = signal_upsampled + noise

        # Plot both waveforms
        fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        ax[0].plot(t, signal_upsampled, color='blue')
        ax[0].set_title("Transmitted BPSK Signal")
        ax[0].set_ylabel("Amplitude")

        ax[1].plot(t, received, color='red')
        ax[1].set_title("Received Signal with Noise")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Amplitude")
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Signal visualization currently supports only BPSK modulation.")


def simulate_traffic(G, intensity, modulation, snr_db):
    nodes = list(G.nodes())
    src = np.random.choice(nodes)
    dst = np.random.choice([n for n in nodes if n != src])
    
    try:
        path = nx.shortest_path(G, source=src, target=dst, weight='latency')
        path_edges = list(zip(path[:-1], path[1:]))
        total_latency = sum(G[u][v]['latency'] for u, v in path_edges)
        min_bandwidth = min(G[u][v]['bandwidth'] for u, v in path_edges)

        st.success(f"Traffic routed from {src} to {dst}")
        st.write(f"Path: {' → '.join(path)}")
        st.write(f"Total Latency: {total_latency} ms")
        st.write(f"Bottleneck Bandwidth: {min_bandwidth} Gbps")

        ber = calculate_ber(modulation, snr_db)
        packet_loss = ber * 100
        st.write(f"Estimated Packet Loss due to BER: {packet_loss:.2f}%")

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, ax=ax, width=1, edge_color='gray')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='red', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[src, dst], node_size=1000, node_color=['green', 'blue'], ax=ax)
        st.pyplot(fig)
        
    except nx.NetworkXNoPath:
        st.error(f"No path exists between {src} and {dst}")
    visualize_signal_waveforms(modulation, snr_db)

def simulate_link_failure(G, pos):
    if len(G.edges()) == 0:
        st.warning("No edges to fail in this topology")
        return
    
    edges = list(G.edges())
    failed_edge = edges[np.random.randint(0, len(edges))]
    u, v = failed_edge
    original_latency = G[u][v]['latency']
    original_bandwidth = G[u][v]['bandwidth']
    
    G[u][v]['latency'] = 999
    G[u][v]['bandwidth'] = 0
    
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
    
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, ax=ax, width=1, edge_color='gray')
    nx.draw_networkx_edges(G, pos, edgelist=[failed_edge], width=3, edge_color='black', style='dashed', ax=ax)

    for src, dst, path in paths_affected[:3]:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2, edge_color='orange', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[src, dst], node_size=1000, node_color=['green', 'blue'], ax=ax)
    
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)
    st.warning(f"Link between {u} and {v} has failed!")
    st.pyplot(fig)

    if paths_affected:
        st.subheader("Protection Switching Activated")
        st.write(f"Failed Link: {u} ↔ {v}")
        st.write(f"Affected Paths: {len(paths_affected)}")
        st.write("Example rerouted paths:")
        for src, dst, path in paths_affected[:3]:
            st.write(f"{src} → {dst}: {' → '.join(path)}")
    else:
        st.info("No active paths affected by this link failure")
    
    G[u][v]['latency'] = original_latency
    G[u][v]['bandwidth'] = original_bandwidth

def modulation_visualization(modulation, snr_db):
    st.header("Modulation Process Visualization")

    # Generate random bits
    bits = np.random.randint(0, 2, 100)

    if modulation == "BPSK":
        # BPSK: 0 → -1, 1 → +1
        symbols = 2 * bits - 1
        t = np.linspace(0, 1, len(symbols))
        modulated = symbols
    elif modulation == "QPSK":
        bits = bits[:len(bits) // 2 * 2]  # Ensure even number of bits for pairing
        bit_pairs = bits.reshape(-1, 2)
        mapping = {
            (0, 0): 1 + 1j,
            (0, 1): -1 + 1j,
            (1, 0): 1 - 1j,
            (1, 1): -1 - 1j,
        }
        symbols = np.array([mapping[tuple(b)] for b in bit_pairs])
        modulated = symbols
        t = np.linspace(0, 1, len(symbols))
    elif modulation == "16-QAM":
        bits = bits[:len(bits) // 4 * 4]  # Ensure the number of bits is a multiple of 4
        symbols = []
        for i in range(0, len(bits), 4):
            I = 2 * (2 * bits[i] + bits[i + 1]) - 3  # Mapping for I axis
            Q = 2 * (2 * bits[i + 2] + bits[i + 3]) - 3  # Mapping for Q axis
            symbols.append(complex(I, Q))
        symbols = np.array(symbols)
        modulated = symbols
        t = np.linspace(0, 1, len(symbols))
    else:
        st.error("Unknown modulation type")
        return

    # Add noise based on SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1 / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), modulated.shape)
    received = modulated + noise

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))

    # Plot Input Bitstream
    axs[0].step(range(len(bits)), bits, where='post')
    axs[0].set_title("Input Bitstream")
    axs[0].set_ylabel("Bits")
    
    # Plot Modulated Signal
    if modulation == "BPSK":
        axs[1].plot(t, modulated)
    else:
        axs[1].plot(t, np.real(modulated), label="Real Part")
        axs[1].plot(t, np.imag(modulated), label="Imaginary Part")
        axs[1].legend()
    axs[1].set_title(f"{modulation} Modulated Signal")
    axs[1].set_ylabel("Amplitude")

    # Plot Received Signal (with noise)
    if modulation == "BPSK":
        axs[2].plot(t, received)
    else:
        axs[2].plot(t, np.real(received), label="Real Part")
        axs[2].plot(t, np.imag(received), label="Imaginary Part")
        axs[2].legend()
    axs[2].set_title("Received Signal (with noise)")
    axs[2].set_ylabel("Amplitude")

    # Demodulated Bitstream (idealized)
    if modulation == "BPSK":
        demodulated = (received > 0).astype(int)
    elif modulation == "QPSK":
        demodulated = np.array([0 if np.real(symbol) > 0 else 1 for symbol in received])
    elif modulation == "16-QAM":
        demodulated = np.array([0 if np.real(symbol) > 0 else 1 for symbol in received])
    else:
        demodulated = np.zeros_like(bits)

    axs[3].step(range(len(bits)), demodulated, where='post')
    axs[3].set_title("Demodulated Bitstream (idealized)")
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel("Bits")

    # Add grid and tighten layout
    for ax in axs:
        ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
