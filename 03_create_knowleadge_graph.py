import json
import pandas as pd
from pathlib import Path
import sys
import networkx as nx
import matplotlib.pyplot as plt

# Define the folder path exactly as you spelled it
folder_path = Path("Assignment/financial_results")
all_tech_records = []
folder_path

print(f"Searching for JSON files in '{folder_path}'...")

# Use .glob() to find all .json files
json_files = list(folder_path.glob("*.json"))


# 2. Iterate through all found JSON files
for file_path in json_files:
    print(f"  -> Reading file: {file_path.name}")
    try:
        # 3. Read and parse the file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Standardize the data: handle if the file contains a single report (dict) or a list of reports
        report_list = []
        if isinstance(data, list):
            report_list = data
        elif isinstance(data, dict):
            report_list = [data] # Treat a single dict as a list with one item
        
        # 4. Extract technologies from each report in the file
        for report in report_list:
            # Get context from the report to add to each technology entry
            company = report.get('company_name', 'Unknown')
            symbol = report.get('symbol', 'Unknown')
            year = report.get('year', 'Unknown')
            quarter = report.get('quarter', 'Unknown')
            
            technologies = report.get('technologies', [])
            
            # 5. Append each technology with its context
            for tech in technologies:
                tech_record = tech.copy()
                tech_record['source_file'] = file_path.name
                tech_record['company_name'] = company
                tech_record['symbol'] = symbol
                tech_record['year'] = year
                tech_record['quarter'] = quarter
                all_tech_records.append(tech_record)

    except json.JSONDecodeError:
        print(f"    - Skipping {file_path.name}: File is not valid JSON.")
    except Exception as e:
        print(f"    - Skipping {file_path.name}: An error occurred: {e}")

# 6. Consolidate into a single DataFrame
if all_tech_records:
    df_all_technologies = pd.DataFrame(all_tech_records)
    
    # Reorder columns to put the most important context first
    context_cols = ['source_file', 'company_name', 'symbol', 'year', 'quarter', 'technology_name']
    other_cols = [col for col in df_all_technologies.columns if col not in context_cols]
    # Ensure all context columns exist before trying to order them
    final_cols_ordered = [col for col in context_cols if col in df_all_technologies.columns] + other_cols
    
    df_all_technologies = df_all_technologies[final_cols_ordered]

    output_file = "all_technologies_compiled.csv"
    df_all_technologies.to_csv(output_file, index=False)
    
    print(f"\n--- Process Complete ---")
    print(f"Successfully extracted and compiled {len(df_all_technologies)} technology entries.")
    print(f"Data saved to '{output_file}'")
    print("\nPreview of the first 5 entries:")
    print(df_all_technologies.head())
else:
    print("\n--- Process Complete ---")
    print("No 'technologies' data was found in any of the processed files.")
    
df_all_technologies.columns

df_all_technologies['technology_name']




# --- 1. Load Data ---


all_tech_df = df_all_technologies[['technology_name','company_name', 'symbol']]

all_tech_df = all_tech_df.dropna(subset=['company_name', 'technology_name'])

#=================================================================
#change the technology from the json


# --- 1. Load the JSON file ---
try:
    with open('Assignment/tech_classification.json', 'r') as f:
        tech_data = json.load(f)
except FileNotFoundError:
    print("Error: 'tech_classification.json' not found.")
    print("Please make sure the file is in the same directory as your script.")

except json.JSONDecodeError:
    print("Error: Could not decode the JSON file. It might be corrupted.")


# 3. Prepare data for the DataFrame

data_list = []

tech_map = {}

# We iterate over the dictionary's items
for main_label, sub_technologies_list in tech_data.items():
    for sub_tech in sub_technologies_list:
        # We use .lower() to ensure matching is case-insensitive
        tech_map[sub_tech.lower()] = main_label

original_column_lowercase = df_all_technologies['technology_name'].str.lower()

# Use .map() to create the new 'techonology' column

df_all_technologies['technology'] = original_column_lowercase.map(tech_map)


# We'll fill 'NaN' values in the 'new_label' column with their original 'technology_name'
df_all_technologies = df_all_technologies[['company_name', 'symbol','technology']].dropna()
df_all_technologies = df_all_technologies.dropna(subset=['technology'])


#================================================================
#building graph
CompanyTechGraph = nx.Graph()


# --- 3. Add 'Company' nodes ---

company_df = df_all_technologies[['company_name', 'symbol']].drop_duplicates()
print(f"Adding {len(company_df)} 'Company' nodes...")


for _, row in company_df.iterrows():
    CompanyTechGraph.add_node(
        row['company_name'],
        type='Company',
        symbol=row['symbol']
    )

# --- 4. Add 'Technology' nodes ---

tech_nodes = df_all_technologies['technology'].unique()
print(f"Adding {len(tech_nodes)} 'Technology' nodes...")

for tech_name in tech_nodes:
    CompanyTechGraph.add_node(
        tech_name,
        type='Technology'
    )

# --- 5. Add 'MENTIONS' edges ---

print(f"Adding {len(df_all_technologies)} 'MENTIONS' edges...")

for _, row in df_all_technologies.iterrows():
    CompanyTechGraph.add_edge(
        row['company_name'],
        row['technology'],
        type='MENTIONS',
        sentiment=row['technology_sentiment'],
        investment=row['investment_action'],
        profit_outlook=row['profit_outlook'],
        mentioned_by=row['mentioned_by'],
        year=row['year'],
        quarter=row['quarter']
    )

# --- 6. Print Summary ---
print(f"\n✅ Company-Technology Graph built successfully!")
print(f"Total nodes: {CompanyTechGraph.number_of_nodes():,}")
print(f"Total edges: {CompanyTechGraph.number_of_edges():,}")

# You can filter nodes to verify the two sets
company_nodes = [n for n, d in CompanyTechGraph.nodes(data=True) if d['type'] == 'Company']
tech_nodes = [n for n, d in CompanyTechGraph.nodes(data=True) if d['type'] == 'Technology']

print(f"  - Company nodes found: {len(company_nodes):,}")
print(f"  - Technology nodes found: {len(tech_nodes):,}")


#==========================================================================
# ========== PLOTING


# --- 1. Create a Sample CompanyTechGraph ---
# (Using the same sample graph as before)
G = CompanyTechGraph


# --- 2. Create a Color Map ---
# (This remains the same)
color_map = []
for node, data in G.nodes(data=True):
    if data.get('type') == 'Company':
        color_map.append('skyblue')  # Color for companies
    elif data.get('type') == 'Technology':
        color_map.append('lightgreen')  # Color for technologies
    else:
        color_map.append('grey')
print("Color map generated.")

# --- 3. Create Custom Labels (The Solution) ---
# Create a dictionary where keys are node names and values are the labels.
# We will ONLY add the technology nodes to this dictionary.
custom_labels = {}
for node, data in G.nodes(data=True):
    if data.get('type') == 'Technology':
        custom_labels[node] = node
        
print(f"Custom label dictionary created. It contains {len(custom_labels)} labels.")

# --- 4. Plot the Graph ---
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, k=0.35, iterations=40, seed=42)

# Draw the graph
nx.draw(G,
        pos,
        node_color=color_map,
        labels=custom_labels,  # <-- Use our custom label dictionary
        node_size=800,
        font_size=10,
        font_weight='bold',
        edge_color='gray',
        width=0.5
       )
# Note: We removed 'with_labels=True' and replaced it with 'labels=custom_labels'

plt.title("Company-Technology Graph (Tech Labels Only)", fontsize=20)
plt.savefig('company_tech_graph_tech_only.png', dpi=300, bbox_inches='tight')

print("Graph plot saved as 'company_tech_graph_tech_only.png'")


#===============================================================================
# Perform centrality analysis


# --- 1. Create a Sample CompanyTechGraph ---
# (Using the same sample graph as before)
G = CompanyTechGraph


# --- 2. Calculate Centrality Measures ---

# Degree Centrality (Normalized)
# This is fast.
print("Calculating Degree Centrality...")
deg_centrality = nx.degree_centrality(G)

# Betweenness Centrality (Normalized)
# *** WARNING: This is VERY SLOW on large graphs. ***
print("Calculating Betweenness Centrality...")
# For your real 1500-node graph, you might want to use a sample:
# between_centrality = nx.betweenness_centrality(G, k=500, normalized=True)
between_centrality = nx.betweenness_centrality(G, normalized=True)

# Eigenvector Centrality
# This can sometimes fail on complex graphs, so we use max_iter
# This is reasonably fast.
print("Calculating Eigenvector Centrality...")
try:
    eigen_centrality = nx.eigenvector_centrality(G, max_iter=1000)
except nx.PowerIterationFailedConvergence:
    print("Eigenvector Centrality did not converge. Skipping.")
    eigen_centrality = {node: 0 for node in G.nodes()} # Create empty dict

print("Centrality calculations complete.")

# --- 3. Combine Measures into a DataFrame ---

# Create a list of dictionaries, one for each node
node_data = []
for node in G.nodes():
    data = G.nodes[node]
    node_data.append({
        'node': node,
        'type': data.get('type'),
        'degree': deg_centrality.get(node, 0),
        'betweenness': between_centrality.get(node, 0),
        'eigenvector': eigen_centrality.get(node, 0)
    })

# Create the DataFrame
df_centrality = pd.DataFrame(node_data)

print("\n--- Full Centrality DataFrame ---")
print(df_centrality)

# --- 4. Find the Most Important Nodes ---

# Separate into Companies and Technologies
df_companies = df_centrality[df_centrality['type'] == 'Company'].copy()
df_techs = df_centrality[df_centrality['type'] == 'Technology'].copy()

print("\n--- Top 3 Companies by Degree (Most Connected) ---")
print(df_companies.sort_values(by='degree', ascending=False).head(3))

print("\n--- Top 3 Technologies by Degree (Most Used) ---")
print(df_techs.sort_values(by='degree', ascending=False).head(3))

print("\n--- Top 3 Nodes by Betweenness (Best 'Brokers') ---")
print(df_centrality.sort_values(by='betweenness', ascending=False).head(3))

print("\n--- Top 3 Nodes by Eigenvector (Most 'Influential') ---")
print(df_centrality.sort_values(by='eigenvector', ascending=False).head(3))


#===============================================================================
#proyecction


# --- Get the Two Node Sets ---

company_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'Company']
tech_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'Technology']

print(f"Found {len(company_nodes)} company nodes and {len(tech_nodes)} technology nodes.")

# --- Create the Weighted Projections ---

CompanyCompanyGraph = nx.bipartite.weighted_projected_graph(G, company_nodes)


print("Creating Technology-Technology projection...")
TechTechGraph = nx.bipartite.weighted_projected_graph(G, tech_nodes)

print("Projections created.")

# --- 4. Compare and Analyze the Projections ---

print("\n--- Analysis: Top Company 'Peers' (Most Shared Tech) ---")

company_edges = sorted(CompanyCompanyGraph.edges(data=True),
                       key=lambda t: t[2].get('weight', 0),
                       reverse=True)

if not company_edges:
    print("No shared technologies found between any companies.")
else:
    for u, v, data in company_edges[:5]: # Show top 5
        print(f"  - {u} <-> {v}: {data['weight']} shared tech(s)")

print("\n--- Analysis: Top 'Tech Bundles' (Most Shared Companies) ---")
# Sort edges by weight, descending
tech_edges = sorted(TechTechGraph.edges(data=True),
                    key=lambda t: t[2].get('weight', 0),
                    reverse=True)

if not tech_edges:
    print("No co-occurring technologies found.")
else:
    for u, v, data in tech_edges[:5]: # Show top 5
        print(f"  - {u} <-> {v}: {data['weight']} shared company(s)")

#==========================================================================
#quarter analysis graph


print(df_all_technologies)

df_all_technologies_Q1 = df_all_technologies[df_all_technologies["quarter"]==1]
df_all_technologies_Q4 = df_all_technologies[df_all_technologies["quarter"]==4]


#===============================
#quarter 1

#building graph
CompanyTechGraph_Q1 = nx.Graph()



company_df = df_all_technologies_Q1[['company_name', 'symbol']].drop_duplicates()
print(f"Adding {len(company_df)} 'Company' nodes...")


for _, row in company_df.iterrows():

    CompanyTechGraph_Q1.add_node(
        row['company_name'],
        type='Company',
        symbol=row['symbol']
    )



tech_nodes = df_all_technologies_Q1['technology'].unique()
print(f"Adding {len(tech_nodes)} 'Technology' nodes...")

for tech_name in tech_nodes:
    CompanyTechGraph_Q1.add_node(
        tech_name,
        type='Technology'
    )


print(f"Adding {len(df_all_technologies_Q1)} 'MENTIONS' edges...")

for _, row in df_all_technologies_Q1.iterrows():
    CompanyTechGraph_Q1.add_edge(
        row['company_name'],
        row['technology'],
        type='MENTIONS',
        sentiment=row['technology_sentiment'],
        investment=row['investment_action'],
        profit_outlook=row['profit_outlook'],
        mentioned_by=row['mentioned_by'],
        year=row['year'],
        quarter=row['quarter']
    )

# --- 6. Print Summary ---
print(f"\n✅ Company-Technology Graph built successfully!")
print(f"Total nodes: {CompanyTechGraph_Q1.number_of_nodes():,}")
print(f"Total edges: {CompanyTechGraph_Q1.number_of_edges():,}")


company_nodes = [n for n, d in CompanyTechGraph_Q1.nodes(data=True) if d['type'] == 'Company']
tech_nodes = [n for n, d in CompanyTechGraph_Q1.nodes(data=True) if d['type'] == 'Technology']

print(f"  - Company nodes found: {len(company_nodes):,}")
print(f"  - Technology nodes found: {len(tech_nodes):,}")


G = CompanyTechGraph_Q1


color_map = []
for node, data in G.nodes(data=True):
    if data.get('type') == 'Company':
        color_map.append('skyblue')  
    elif data.get('type') == 'Technology':
        color_map.append('lightgreen')  
    else:
        color_map.append('grey')
print("Color map generated.")


custom_labels = {}
for node, data in G.nodes(data=True):
    if data.get('type') == 'Technology':
        custom_labels[node] = node
        
print(f"Custom label dictionary created. It contains {len(custom_labels)} labels.")


plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, k=0.35, iterations=40, seed=42)


nx.draw(G,
        pos,
        node_color=color_map,
        labels=custom_labels,  
        node_size=800,
        font_size=10,
        font_weight='bold',
        edge_color='gray',
        width=0.5
       )

plt.title("Graph Visualization for Q1", fontsize=20)


plt.savefig('company_tech_graph_Q1.png', dpi=300, bbox_inches='tight')

#==================================================================================
#centrality measures



print("Calculating Degree Centrality...")
deg_centrality = nx.degree_centrality(G)


between_centrality = nx.betweenness_centrality(G, normalized=True)

print("Calculating Eigenvector Centrality...")
try:
    eigen_centrality = nx.eigenvector_centrality(G, max_iter=1000)
except nx.PowerIterationFailedConvergence:
    print("Eigenvector Centrality did not converge. Skipping.")
    eigen_centrality = {node: 0 for node in G.nodes()} # Create empty dict

print("Centrality calculations complete.")

# ---Combine Measures into a DataFrame ---

node_data = []
for node in G.nodes():
    data = G.nodes[node]
    node_data.append({
        'node': node,
        'type': data.get('type'),
        'degree': deg_centrality.get(node, 0),
        'betweenness': between_centrality.get(node, 0),
        'eigenvector': eigen_centrality.get(node, 0)
    })

# Create the DataFrame
df_centrality = pd.DataFrame(node_data)

print("\n--- Full Centrality DataFrame ---")
print(df_centrality)

# --- 4. Find the Most Important Nodes ---


df_companies = df_centrality[df_centrality['type'] == 'Company'].copy()
df_techs = df_centrality[df_centrality['type'] == 'Technology'].copy()

print("\n--- Top 3 Companies by Degree (Most Connected) ---")
print(df_companies.sort_values(by='degree', ascending=False).head(3))

print("\n--- Top 3 Technologies by Degree (Most Used) ---")
print(df_techs.sort_values(by='degree', ascending=False).head(3))

print("\n--- Top 3 Nodes by Betweenness (Best 'Brokers') ---")
print(df_centrality.sort_values(by='betweenness', ascending=False).head(3))

print("\n--- Top 3 Nodes by Eigenvector (Most 'Influential') ---")
print(df_centrality.sort_values(by='eigenvector', ascending=False).head(3))


#===============================
#quarter 4

#building graph
CompanyTechGraph_Q4 = nx.Graph()



company_df = df_all_technologies_Q4[['company_name', 'symbol']].drop_duplicates()
print(f"Adding {len(company_df)} 'Company' nodes...")


for _, row in company_df.iterrows():
 
    CompanyTechGraph_Q4.add_node(
        row['company_name'],
        type='Company',
        symbol=row['symbol']
    )



tech_nodes = df_all_technologies_Q4['technology'].unique()
print(f"Adding {len(tech_nodes)} 'Technology' nodes...")

for tech_name in tech_nodes:
    CompanyTechGraph_Q4.add_node(
        tech_name,
        type='Technology'
    )


print(f"Adding {len(df_all_technologies_Q4)} 'MENTIONS' edges...")

for _, row in df_all_technologies_Q4.iterrows():
    CompanyTechGraph_Q4.add_edge(
        row['company_name'],
        row['technology'],
        type='MENTIONS',
        sentiment=row['technology_sentiment'],
        investment=row['investment_action'],
        profit_outlook=row['profit_outlook'],
        mentioned_by=row['mentioned_by'],
        year=row['year'],
        quarter=row['quarter']
    )

# --- 6. Print Summary ---
print(f"\n✅ Company-Technology Graph built successfully!")
print(f"Total nodes: {CompanyTechGraph_Q4.number_of_nodes():,}")
print(f"Total edges: {CompanyTechGraph_Q4.number_of_edges():,}")


company_nodes = [n for n, d in CompanyTechGraph_Q4.nodes(data=True) if d['type'] == 'Company']
tech_nodes = [n for n, d in CompanyTechGraph_Q4.nodes(data=True) if d['type'] == 'Technology']

print(f"  - Company nodes found: {len(company_nodes):,}")
print(f"  - Technology nodes found: {len(tech_nodes):,}")


G = CompanyTechGraph_Q4

# --- 2. Create a Color Map ---
color_map = []
for node, data in G.nodes(data=True):
    if data.get('type') == 'Company':
        color_map.append('skyblue')  # Color for companies
    elif data.get('type') == 'Technology':
        color_map.append('lightgreen')  # Color for technologies
    else:
        color_map.append('grey')
print("Color map generated.")


custom_labels = {}
for node, data in G.nodes(data=True):
    if data.get('type') == 'Technology':
        custom_labels[node] = node
        
print(f"Custom label dictionary created. It contains {len(custom_labels)} labels.")

# --- 4. Plot the Graph ---
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, k=0.35, iterations=40, seed=42)

# Draw the graph
nx.draw(G,
        pos,
        node_color=color_map,
        labels=custom_labels,  # <-- Use our custom label dictionary
        node_size=800,
        font_size=10,
        font_weight='bold',
        edge_color='gray',
        width=0.5
       )

plt.title("Graph Visualization for Q4", fontsize=20)


plt.savefig('company_tech_graph_Q4.png', dpi=300, bbox_inches='tight')

#===========

print("Calculating Degree Centrality...")
deg_centrality = nx.degree_centrality(G)


print("Calculating Betweenness Centrality...")

between_centrality = nx.betweenness_centrality(G, normalized=True)


print("Calculating Eigenvector Centrality...")
try:
    eigen_centrality = nx.eigenvector_centrality(G, max_iter=1000)
except nx.PowerIterationFailedConvergence:
    print("Eigenvector Centrality did not converge. Skipping.")
    eigen_centrality = {node: 0 for node in G.nodes()} # Create empty dict

print("Centrality calculations complete.")


node_data = []
for node in G.nodes():
    data = G.nodes[node]
    node_data.append({
        'node': node,
        'type': data.get('type'),
        'degree': deg_centrality.get(node, 0),
        'betweenness': between_centrality.get(node, 0),
        'eigenvector': eigen_centrality.get(node, 0)
    })

df_centrality = pd.DataFrame(node_data)

print("\n--- Full Centrality DataFrame ---")
print(df_centrality)


df_companies = df_centrality[df_centrality['type'] == 'Company'].copy()
df_techs = df_centrality[df_centrality['type'] == 'Technology'].copy()

print("\n--- Top 3 Companies by Degree (Most Connected) ---")
print(df_companies.sort_values(by='degree', ascending=False).head(3))

print("\n--- Top 3 Technologies by Degree (Most Used) ---")
print(df_techs.sort_values(by='degree', ascending=False).head(3))

print("\n--- Top 3 Nodes by Betweenness (Best 'Brokers') ---")
print(df_centrality.sort_values(by='betweenness', ascending=False).head(3))

print("\n--- Top 3 Nodes by Eigenvector (Most 'Influential') ---")
print(df_centrality.sort_values(by='eigenvector', ascending=False).head(3))
