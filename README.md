# PRIMO: Progressive Induction for Multi-hop Open Rule Generation

This repository contains the implementation of **PRIMO**, a progressive multi-stage method for multi-hop open rule generation. The method, introduced in the paper *PRIMO: Progressive Induction for Multi-hop Open Rule Generation* (LREC-COLING 2024), aims to improve the quality, diversity, and logical consistency of generated multi-hop open rules.

## Background

Open rules refer to logical deductions that map premise atoms to hypothesis atoms, capturing relationships between entities in natural language. Existing methods for open rule generation face several limitations:
1. **Single-hop limitation**: Existing models focus on single-hop rule generation, limiting their applicability to complex scenarios.
2. **Logical inconsistencies**: Multi-hop generation often introduces contradictions between premise and hypothesis atoms.
3. **Semantic repetition**: Generated rule atoms tend to lack diversity, resulting in redundant outputs.

To address these issues, PRIMO introduces a **progressive multi-stage framework** consisting of three modules:
- **Generation**: Creates descriptive text for hypothesis atoms based on premise atoms.
- **Extraction**: Extracts hypothesis atoms from the text generated in the first stage.
- **Ranking**: Evaluates and ranks candidate hypothesis atoms to ensure logical consistency and diversity.
- 
## Methodology

### Key Features
- **Ontology Integration**: Incorporates entity type information into prompts to enhance rule accuracy and reduce ambiguity.
- **Progressive Framework**: Utilizes generation, extraction, and ranking modules to sequentially refine rule generation.
- **Reinforcement Learning from Human Feedback (RLHF)**: Fine-tunes the model with human feedback to optimize generation quality.

## Evaluation

### Dataset
The paper introduces a benchmark dataset containing 495 premise atoms and 2851 samples. Each sample includes:
- Premise atoms
- Entity types
- Open rule chains with 1 to 5 hops

### Results
| Our Dataset             | B1   | B2   | B4   | RL   | Self-B2 |
|-------------------------|------|------|------|------|---------|
| **Prompt**              | 33.0 | 13.2 | 0.1  | 39.2 | 89.2    |
| **COMET**               | 35.1 | 13.6 | 1.1  | 42.9 | 92.6    |
| **Orion**               | 39.9 | **19.5** | 0.1  | 52.5 | 86.4    |
| **Vicuna-13B**          | **44.8** | 17.4 | **2.9** | **67.9** | 75.5    |
| **PRIMO**               | 44.3 | 16.5 | 2.1  | 66.3 | 80.5    |
| **PRIMO-without RLHF**  | 42.5 | 15.0 | 2.0  | 64.5 | 77.7    |
| **PRIMO-train G_Net**   | 43.4 | 14.1 | 1.3  | 66.3 | 77.8    |
| **PRIMO-train E_Net**   | 40.7 | 15.1 | 2.4  | 62.1 | **70.7** |

## Files and Directories
* `trl_rule/`: Contains functions that perform reinforcement learning on the model.
* `dataset/`: Contains corpus for model training and benchmarks for testing.
* `gen_openrule`: Generates a multi-hop open rule. 


## Citation
If you find this work useful, please cite:
```
@inproceedings{liu2024primo,
  title={PRIMO: Progressive Induction for Multi-hop Open Rule Generation},
  author={Liu, Jianyu and Bi, Sheng and Qi, Guilin},
  booktitle={Proceedings of LREC-COLING 2024},
  year={2024},
  pages={12988--12998}
}
```

## Acknowledgements
PRIMO is indebted to the following outstanding open-source projects:
- [Orion](https://github.com/chenxran/Orion)
- [RLHF](https://github.com/HarderThenHarder/transformers_tasks/tree/main/RLHF)



