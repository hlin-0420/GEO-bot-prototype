from pyvis.network import Network
import networkx as nx
from bs4 import BeautifulSoup
import webbrowser
import os

def build_knowledge_graph_from_table(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        G = nx.Graph()

        for table in soup.find_all('table'):
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if len(cells) == 2:
                    entity, limit = cells[0].get_text(strip=True), cells[1].get_text(strip=True)
                    if entity and limit:
                        G.add_node(entity, label='Type')
                        G.add_node(limit, label='Limit')
                        G.add_edge(entity, limit)

        return G

def visualize_graph_pyvis(G):
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
    net.barnes_hut()

    for node, data in G.nodes(data=True):
        net.add_node(node, label=node, title=f"Type: {data.get('label', 'N/A')}", color='lightgreen')

    for source, target in G.edges():
        net.add_edge(source, target)

    output_filename = "output/knowledge_graph_geo_limits.html"
    output_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), output_filename)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    net.write_html(output_dir)
    webbrowser.open("file://" + output_dir)
    print(f"Interactive graph saved as {output_dir}")

if __name__ == "__main__":
    html_file_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/GEO_Limits.htm')
    G = build_knowledge_graph_from_table(html_file_path)
    visualize_graph_pyvis(G)