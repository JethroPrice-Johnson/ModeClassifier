import streamlit as st
import ezdxf
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import pickle

st.set_page_config(layout="wide")

# Get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTING_FOLDER = os.path.join(SCRIPT_DIR, 'testing_sections')  # Folder containing test DXF files

# Label mappings
LABEL_MAPPING = {
    'uce': 'Unstiffened Compression Element',
    'sce': 'Stiffened Compression Element',
    '1is': '1 Intermediate Stiffener',
    '2is': '2 Intermediate Stiffeners',
    '3is': '3 Intermediate Stiffeners'
}

# Helper function to calculate the angle between two vectors
def calculate_angle(start1, end1, start2, end2):
    vec1 = np.array([end1[0] - start1[0], end1[1] - start1[1]])
    vec2 = np.array([end2[0] - start2[0], end2[1] - start2[1]])
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    unit_vec1 = vec1 / norm1
    unit_vec2 = vec2 / norm2
    dot_product = np.dot(unit_vec1, unit_vec2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product) * (180 / np.pi)
    return angle

# Extract groups and their relationships from DXF
def extract_groups_and_relationships(file_path):
    doc = ezdxf.readfile(file_path)
    modelspace = doc.modelspace()

    group_elements = []
    all_entities = []

    for name, group in doc.groups:
        group_entities = []
        group_connections = defaultdict(list)
        try:
            for entity in group:
                if entity.dxftype() == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    group_entities.append({
                        'start': start,
                        'end': end,
                        'length': np.linalg.norm(np.array(end) - np.array(start)),
                        'entity': entity
                    })
                    group_connections[start].append(entity)
                    group_connections[end].append(entity)
        except Exception as e:
            st.warning(f"Error processing group '{name}': {e}")
            continue

        if group_entities:
            group_elements.append({
                'entities': group_entities,
                'connections': group_connections,
                'group': group,
                'name': name
            })

    for entity in modelspace:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            all_entities.append({
                'start': start,
                'end': end,
                'entity': entity
            })

    return group_elements, all_entities

# Analyze group relationships and add additional features
def analyze_group_relationships(group_element):
    entities = group_element['entities']
    connections = group_element['connections']

    total_length = sum(e['length'] for e in entities)
    num_entities = len(entities)
    num_unique_nodes = len(connections)
    avg_length = total_length / num_entities if num_entities > 0 else 0

    angles = []
    for node, connected_entities in connections.items():
        if len(connected_entities) >= 2:
            for i in range(len(connected_entities)):
                for j in range(i+1, len(connected_entities)):
                    entity1 = connected_entities[i]
                    entity2 = connected_entities[j]
                    start1 = entity1.dxf.start
                    end1 = entity1.dxf.end
                    start2 = entity2.dxf.start
                    end2 = entity2.dxf.end
                    angle = calculate_angle(start1, end1, start2, end2)
                    angles.append(angle)
    avg_angle = np.mean(angles) if angles else 0

    features = [
        total_length,
        num_entities,
        num_unique_nodes,
        avg_length,
        avg_angle
    ]

    group_element['features'] = features

    return group_element

# Visualize the classification result on the DXF file drawing
def visualize_classification_of_groups(group_elements, classifications, figsize=(15, 15)):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    unique_classes = list(set(classifications))
    class_to_color = {class_name: colors[i % len(colors)] for i, class_name in enumerate(unique_classes)}

    # Set up scaling based on figure size
    base_figsize = 10  # Reference figure size
    scaling_factor = figsize[0] / base_figsize

    line_width = 6 * scaling_factor
    font_size_title = 16 * scaling_factor
    font_size_legend = 12 * scaling_factor

    # Create the figure with the specified size
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each group with the specified line width
    for i, group_element in enumerate(group_elements):
        group_entities = group_element['entities']
        class_label = classifications[i]
        color = class_to_color[class_label]

        for element in group_entities:
            start = element['start']
            end = element['end']
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=line_width)

    # Set the title with the specified font size
    ax.set_title("Classified Elements", fontsize=font_size_title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')
    ax.grid(False)  # Remove the grid

    # Remove the axes and border spines
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Remove ticks
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    # Create a legend with the scaled font size
    handles = [plt.Line2D([0], [0], color=class_to_color[class_name], lw=line_width, label=LABEL_MAPPING.get(class_name, class_name)) for class_name in unique_classes]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1), fontsize=font_size_legend)

    return fig

def main():
    st.title("DXF Group Classification App")

    # Paths to folders
    temp_folder = os.path.join(SCRIPT_DIR, 'temp')
    os.makedirs(temp_folder, exist_ok=True)  # Ensure the temp folder exists
    os.makedirs(TESTING_FOLDER, exist_ok=True)  # Ensure the testing_sections folder exists

    # Get a list of DXF files in the testing folder
    dxf_files = [f for f in os.listdir(TESTING_FOLDER) if f.endswith('.dxf')]

    # Interface with columns
    col1, col2 = st.columns([1, 1])

    with col1:
        # Display dropdown list of DXF files in "testing_sections"
        if not dxf_files:
            st.warning(f"No DXF files found in '{TESTING_FOLDER}'. Please add DXF files to the folder.")
            return
        selected_file = st.selectbox("Select a DXF file to analyze:", dxf_files)

        # Run button
        run_pressed = st.button('Run')

        if run_pressed and selected_file:
            # Set the selected file path
            temp_file_path = os.path.join(TESTING_FOLDER, selected_file)
            st.session_state['temp_file_path'] = temp_file_path
            st.session_state['run_pressed'] = True

    # Handle actions after Run is pressed
    if st.session_state.get('run_pressed', False):
        if 'temp_file_path' not in st.session_state:
            st.warning("Please select a file to proceed.")
            return

        temp_file_path = st.session_state['temp_file_path']

        # Check if a trained model exists
        model_file_path = os.path.join(SCRIPT_DIR, 'trained_model.pkl')
        if os.path.exists(model_file_path):
            # Load the trained model
            with open(model_file_path, 'rb') as f:
                trained_model = pickle.load(f)
            st.session_state['trained_model'] = trained_model

            # Proceed to classify
            group_elements, all_entities = extract_groups_and_relationships(temp_file_path)
            if not group_elements:
                st.warning("No groups found in the selected DXF file.")
                return
            classifications = []
            for group_element in group_elements:
                group_with_relationships = analyze_group_relationships(group_element)
                features = group_with_relationships['features']
                predicted_label = trained_model.predict([features])[0]
                classifications.append(predicted_label)
            # Visualize the results
            fig = visualize_classification_of_groups(group_elements, classifications)
            st.session_state['classification_fig'] = fig  # Store the figure in session state
            st.session_state['run_pressed'] = False  # Reset to prevent re-entering this block
        else:
            st.warning("No trained model found. Please ensure a trained model exists before running classification.")
            return

    # Display the final classified image
    if 'classification_fig' in st.session_state and 'trained_model' in st.session_state:
        with col2:
            st.pyplot(st.session_state['classification_fig'])

if __name__ == "__main__":
    main()
