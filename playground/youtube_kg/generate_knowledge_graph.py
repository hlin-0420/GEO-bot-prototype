from pyvis.network import Network
import networkx as nx
from flair.data import Sentence
from flair.models import SequenceTagger
from extract_transcript import get_transcript
import webbrowser
import os

# Load Flair NER Tagger
tagger = SequenceTagger.load("ner")

def build_knowledge_graph(text_data):
    # Use flair for NER
    sentence = Sentence(text_data)
    tagger.predict(sentence)

    # Extract entities
    entities = [(entity.text, entity.get_label('ner').value) for entity in sentence.get_spans('ner')]

    # Build the graph
    G = nx.Graph()
    for i in range(len(entities) - 1):
        G.add_node(entities[i][0], label=entities[i][1])
        G.add_node(entities[i+1][0], label=entities[i+1][1])
        G.add_edge(entities[i][0], entities[i+1][0])
    
    return G

def visualize_graph_pyvis(G):
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
    net.barnes_hut()

    for node, data in G.nodes(data=True):
        net.add_node(node, label=node, title=f"Type: {data.get('label', 'N/A')}", color='lightgreen')

    for source, target in G.edges():
        net.add_edge(source, target)

    output_folder = "output/knowledge_graph.html"
    
    output_dir = os.path.join(os.path.dirname(__file__), output_folder)
    
    net.write_html(output_dir)
    
    webbrowser.open("file://" + output_dir)
    print(f"Interactive graph saved as {output_dir}")

if __name__ == "__main__":
    video_id = 'cVRuLaEJij0'  # NOTE: Remove '&t=890s' for transcript fetching
    text_data = get_transcript(video_id)
    G = build_knowledge_graph(text_data)
    visualize_graph_pyvis(G)
