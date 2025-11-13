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

3. 02_EDA.ipynb: Exploratory Data Analysis
Performs exploratory data analysis on the JSON files generated in the previous step.

This script generates insights and visualizations about the distribution of technologies and company mentions.

4. 03_create_knowleadge_graph.ipynb: Network Construction
Uses the structured JSON data to build a knowledge graph.

This graph maps the bipartite relationships between Companies and the Technologies they mention, forming the basis for the network analysis.

Key Findings
Analysis of the Company-Technology graph revealed a significant concentration of companies involved in AI & Machine Learning.

This technology category does not exist in isolation; it serves as a central hub connected to other foundational technologies:

Cloud & Data Center Infrastructure: The primary infrastructure that powers AI.

Software, Apps & Digital Platforms: The primary channel that delivers AI-driven features to users.

Data & Analytics: The essential ecosystem that feeds AI models.

Limitations and Future Work
This analysis faced several key limitations:

Technology Labeling: Accurately classifying all mentioned technologies was challenging. Many specific, non-generic, or proprietary technology names were difficult to map to our broader labels.

Scope: Due to these labeling challenges, many specific technologies could not be included in the final analysis, potentially underrepresenting the long tail of tech adoption.

