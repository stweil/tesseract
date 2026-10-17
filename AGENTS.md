# Agents

This document describes how AI agents should interact with this codebase.

## Project Overview

Tesseract is an open source OCR (Optical Character Recognition) engine. It is used to convert images of text into machine-readable text.

## Working with this Codebase

- Follow existing code conventions and style
- Run linting and typechecking before submitting changes when available
- Do not commit secrets or keys to the repository
- Write tests for new functionality

## Codebase Structure

- `src/api/` - Public C++ API
- `src/ccmain/` - Main recognition pipeline
- `src/ccstruct/` - Core data structures (blobs, outlines, page layout)
- `src/ccutil/` - Utilities (memory, file I/O, parameters)
- `src/classify/` - Character classification engine
- `src/dict/` - Dictionary, DAWG, and language modeling
- `src/lstm/` - LSTM-based neural network recognizer
- `src/textord/` - Text layout analysis and page structure
- `src/wordrec/` - Word recognition and segmentation
- `src/training/` - Training tools and utilities
- `src/viewer/` - ScrollView debugging interface
- `unittest/` - Unit tests
- `test/` - Integration tests and test data
- `tessdata/` - Trained data files

## Development Workflow

1. Explore the codebase to understand the structure
2. Make changes following existing patterns
3. Verify changes with available tests
4. Run lint/type checks if configured