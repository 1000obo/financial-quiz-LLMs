# Automated Generation and Evaluation of Financial Quizzes using LLMs

Framework for generating and evaluating financial literacy quizzes in European Portuguese (PT-PT) using financial articles and newsletters.

## Overview

This project implements a two-stage pipeline to generate multiple-choice financial literacy quizzes using Retrieval-Augmented Generation (RAG) and Distractor Theory.



### Features

* **Knowledge Module:** Vector search over 74 articles/newsletters using OpenAI `text-embedding-3-small` stored in ChromaDB, followed by context-bounded summarization via `gpt-4o-mini`.
* **Quiz Module:** Question and correct answer generation using `gpt-4o`, followed by distractor generation employing Distractor Theory rules.
* **Difficulty Levels:** Supports three financial literacy levels: `Baixo` (Low), `Médio` (Medium), and `Alto` (High).
* **Evaluation Pipeline:** Evaluation approach including LLM-as-a-judge scoring, simulated LLM user personas, and human expert assessment.

## Installation and Setup

### Prerequisites

* Python 3.10 or higher
* `uv` or `pip`

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/1000obo/financial-quiz-LLMs.git
   cd code
   ```

2. Create and activate a virtual environment using Python 3.10+:
   ```bash
   uv venv --python 3.10
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

4. Environment Variables:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### 1. Generate Quizzes

Run the generation pipeline (Knowledge Retrieval + Question & Distractor Generation):

```bash
python3 quiz_generation.py
```

Generated quizzes are saved as JSON files in the output directory.

### 2. Evaluate Quizzes

Run the evaluation pipeline (LLM-as-a-judge & Persona simulations):

```bash
python3 quiz_evaluation.py
```

## Dataset Details

* **Knowledge Base:** 74 financial articles and newsletters provided by Paynest covering topics such as credit management, savings, budgeting, interest rates, and taxes.
* **Dataset Generated:** 24 quizzes across 8 topics and 3 literacy levels (240 questions total).
* **Evaluation Subset:** 120 questions evaluated by human experts and LLM judges.