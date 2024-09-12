# Verisynth: Large Language Model Synthesized Access Control Policy Verification

## Overview

This repository contains the code and instructions to replicate our experiments on using large language models for access control policy synthesis and verification. Our study explores the generation and analysis of access control policies using state-of-the-art language models, with a focus on AWS policies.

## Model Selection Rationale

We chose Claude-3.5 Sonnet for our primary policy generation tasks due to its exceptional performance across various LLM evaluation benchmarks, comparable to OpenAI's GPT-4o. In our initial policy comparison experiment, Claude-3.5 Sonnet consistently outperformed GPT-4o in policy-related tasks.

While Mistral Large showed superior performance in our specific policy comparison experiment, its overall capabilities as measured by widely recognized open benchmarks lag behind both GPT-4o and Claude-3.5 Sonnet. We determined that Claude-3.5 Sonnet offered the best balance of general capability and task-specific performance for our subsequent experiments.

## Experimental Approach

During our initial exploratory phase, we experimented with various LLMs, including GPT-3.5, GPT-4, and open-source models like Llama 3 and Llama 3.1 (7B and 42B parameter models). However, due to computational demands and resource limitations, we shifted to using LLM APIs from Mistral, OpenAI, and Anthropic for our empirical study on synthesizing policies.

For our main experiments, we used:
- Claude-3.5 Sonnet for policy generation
- A custom fine-tuned GPT-4o-mini for regex generation

This combination allowed us to leverage the strengths of different models tailored to our specific research objectives.

## Prerequisites

- Python 3.8+
- Anthropic API key
- OpenAI API key (for fine-tuned model experiments)
- Access to AWS policies dataset (or your own dataset of access control policies)
- ABC (Automata-Based model Counter)
- Quacky (Quantitative Access Control Permissiveness Analyzer)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/verisynth.git
   cd verisynth
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   Create a file named `llms.env` in the root directory and add your API keys:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Install ABC (Automata-Based model Counter):
   Follow the installation instructions at:
   https://github.com/vlab-cs-ucsb/ABC

5. Install Quacky (Quantitative Access Control Permissiveness Analyzer):
   Follow the installation instructions at:
   https://github.com/vlab-cs-ucsb/quacky

   Note: Ensure that both ABC and Quacky are properly installed and accessible in your system's PATH.

## Experiments

### Experiment 1: Policy Generation and Comparison

To run the policy generation and comparison experiment:

```
python exp1_policy_generation.py
```

### Experiment 2: Resource Summarization

To run the resource summarization experiment:

```
python exp2_resource_summarization.py
```

This experiment evaluates Verisynth's ability to generate concise and accurate regular expressions (regexes) that summarize the resources allowed by access control policies. It assesses how well Verisynth can abstract and represent complex policy permissions in a compact form, measuring factors such as regex complexity, processing time, and semantic accuracy (via Jaccard similarity).

### Experiment 3: Factors Affecting Summarization Accuracy

To run the experiment on factors affecting summarization accuracy:

```
python exp3_summarization_factors.py
```

This experiment investigates factors influencing the accuracy of resource summarization in access control policies using three approaches: direct simplification, pre-trained generalization, and fine-tuned generalization. It also explores the impact of varying the number of enumerated strings on summarization accuracy.

## Data
- We have made available the entire Dataset used for this project, to add your own:
- Place your AWS policies dataset in the `Dataset` folder.
- Generated policies will be saved in the `generated_policies` folder.
- Results will be saved in CSV format in the `results` folder.

## Replicating Results

To replicate our results:

1. Ensure you have the same dataset of AWS policies used in our study.
2. Run each experiment script as described above.
3. The results will be saved in CSV files in the `results` folder.
4. Use the provided Jupyter notebooks in the `analysis` folder to generate charts and perform statistical analyses.

## Note

Due to the non-deterministic nature of language models, exact replication of results may not be possible. However, you should observe similar trends and patterns in your results.

## Citation

If you use this code or our findings in your research, please cite our paper:

[Citation information will be added upon publication]

## License

[Include your chosen license information here]
```

