# PodcastIQ – REALM-Based Podcast Retrieval Demo

PodcastIQ is a podcast transcript retrieval project that demonstrates **REALM-style retrieval-augmented question answering**.

This repo can be run in two ways:

1. **Local mode**  
   Use a transcript text file on your machine and run retrieval locally.

2. **Snowflake mode**  
   Use Snowflake as the external knowledge store for transcripts/chunks.

This project is useful for showing the core REALM idea:

> Instead of answering directly from model memory, first retrieve relevant external knowledge, then answer using that evidence.

---

# Project Overview

This repo contains:

- a **basic local retrieval version** without OpenAI generation
- an **improved local semantic retrieval version** using OpenAI embeddings + grounded answer generation
- a **Snowflake-based version** for using a database-backed knowledge store

---

# REALM Concept in This Project

In this project, REALM is demonstrated as:

1. **Knowledge source exists outside the model**
   - locally in a transcript text file, or
   - in Snowflake tables

2. **Transcript is split into chunks**

3. **Relevant chunks are retrieved for a user question**

4. **The final answer is generated using only the retrieved chunks**

This makes answers more grounded and explainable.

---

# Repository Structure

```text
PodcastIQ/
│
├── data/
│   ├── sample_transcript.txt
│   └── processed/
│       └── chunks.json
│
├── outputs/
│   ├── retrieval_output.json
│   ├── openai_retrieval_output.json
│   ├── embedded_chunks_cache.json
│   └── embedded_chunks_preview.json
│
├── local_realm_test.py
├── openai_local_realm_test.py
├── requirements.txt
├── .env
└── README.md
