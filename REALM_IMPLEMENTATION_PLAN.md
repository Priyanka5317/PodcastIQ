# REALM Implementation Plan

## Goal
Implement a REALM-style iterative retrieval and grounded generation backend for PodcastIQ using Snowflake Cortex and the existing semantic embeddings table.

## Added components
- Query embedding via Snowflake Cortex
- First-pass retrieval from SEM_CHUNK_EMBEDDINGS
- Evidence sufficiency check via Snowflake Cortex COMPLETE
- Refined query generation via Snowflake Cortex COMPLETE
- Second-pass retrieval
- Final grounded answer generation via Snowflake Cortex COMPLETE

## Milestones
- [ ] Create file structure
- [ ] Verify Snowflake connection
- [ ] Verify Cortex embedding works
- [ ] Verify retrieval works
- [ ] Verify evidence check works
- [ ] Verify refined retrieval works
- [ ] Verify final answer generation works
- [ ] Save outputs for presentation/demo
