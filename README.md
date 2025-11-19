# Financial Records Analysis: Mapping Technology Adoption in the S&P 500
This project analyzes S&P 500 earnings call transcripts from 2024 to extract, analyze, and visualize the relationships between companies and the technologies they discuss,
with a special focus on the penetration of AI.

The data source used is the "Kurry" dataset (S&P 500 earnings transcripts) available on Hugging Face.

Project Pipeline
The analysis is broken down into the following scripts:

1. 00_load_data.py: Data Acquisition and Filtering
Extracts all available earnings call transcripts from 2024.

Selects the first 1,000 records for analysis.

The focus on 2024 data ensures the analysis is based on the most up-to-date information regarding technology and AI adoption.

2. 01_feauture_extraction.ipynb: LLM-based Feature Extraction
This notebook details the pipeline for extracting structured data from large, unstructured transcript texts.

Chunking: Transcripts were too large for direct processing. The langchain-community library was used to split them into smaller, manageable chunks.

Prioritization: A specific query was used to identify the most relevant chunks from each transcript.

Embedding: An Ollama embedding model was used to process the prioritized chunks, reducing their dimensionality.

Extraction: A Gemini model processed the embedded data to generate a structured JSON file containing the attributes of interest (e.g., company name, technologies mentioned).

Output: Folder finantial_results_data

3. 02_EDA.ipynb: Exploratory Data Analysis
Performs exploratory data analysis on the JSON files generated in the previous step.

This script generates insights and visualizations about the distribution of technologies and company mentions.

4. 03_create_knowleadge_graph.ipynb: Network Construction
Uses the structured JSON data to build a knowledge graph.

This graph maps the bipartite relationships between Companies and the Technologies they mention, forming the basis for the network analysis.

## Key Findings

# Network Centrality Analysis: Companies vs. Technologies

Based on the graph analysis output, we can evaluate the structural importance of specific companies and technologies within the network. The data is categorized by **Degree** (connections), **Betweenness** (brokerage ability), and **Eigenvector** (influence).

## 1. Top Companies by Connectivity

The following companies have the highest "Degree" centrality, indicating they have the most direct links in the dataset.

| Rank | Company | Degree | Betweenness | Eigenvector | Website |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Analog Devices, Inc.** | 0.117188 | 0.034876 | 0.133554 | [analog.com](https://www.analog.com) |
| 2 | **Applied Materials, Inc.** | 0.093750 | 0.017646 | 0.137164 | [appliedmaterials.com](https://www.appliedmaterials.com) |
| 3 | **Amphenol Corporation** | 0.070312 | 0.015438 | 0.083062 | [amphenol.com](https://www.amphenol.com) |

## 2. Top Technologies by Network Impact

Technologies show significantly higher centrality scores than individual companies, suggesting they act as the primary hubs of this network. Note that the ranking order shifts slightly between metrics.

| Technology Node | Degree (Most Used) | Betweenness (Best Broker) | Eigenvector (Most Influential) |
| :--- | :--- | :--- | :--- |
| **AI & Machine Learning** | **0.554688 (1st)** | **0.295911 (1st)** | **0.385379 (1st)** |
| **Cloud & Data Center** | 0.437500 (2nd) | 0.161738 (3rd) | 0.307624 (2nd) |
| **Software & Platforms** | 0.398438 (3rd) | 0.171698 (2nd) | 0.281470 (3rd) |

## 3. Analytical Conclusion

The network topology reveals a technology-centric ecosystem driven by three major pillars, with specific hardware companies acting as key facilitators.

### A. The Dominance of AI
**AI & Machine Learning** is the unequivocal center of this network.
* **Score Analysis:** It holds the top score in every category. With a Degree of `0.55`, it is connected to over
Analysis of the Company-Technology graph revealed a significant concentration of companies involved in AI & Machine Learning.

This technology category does not exist in isolation; it serves as a central hub connected to other foundational technologies:

Cloud & Data Center Infrastructure: The primary infrastructure that powers AI.

Software, Apps & Digital Platforms: The primary channel that delivers AI-driven features to users.

Data & Analytics: The essential ecosystem that feeds AI models.

Limitations and Future Work
This analysis faced several key limitations:

Technology Labeling: Accurately classifying all mentioned technologies was challenging. Many specific, non-generic, or proprietary technology names were difficult to map to our broader labels.

Scope: Due to these labeling challenges, many specific technologies could not be included in the final analysis, potentially underrepresenting the long tail of tech adoption.

