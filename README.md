# Soft Prompting for Image Classification (Replication) + VQA (Extension)

Collaborator: Smruti Chourasia

This project intends to investigate the Context Optimization (CoOp) (Zhou et al., 2022b) and Conditional Context Optimization (CoCoOp) (Zhou et al., 2022a) methods. We reimplemented both methods for image classification and extended them to visual question answering (VQA) using native PyTorch & PyTorch Lightning packages. Our experiments indicate 3 major results for both tasks. First, few-shot learning with either method performs better than zero-shot inference from the base model, Contrastive Language-Image Pretraining (CLIP) (Radford et al., 2021). Second, both methods perform better with more shots, indicating a potential tradeoff between inference efficiency and accuracy. Third, CoCoOp performs better than CoOp. We successfully validated the results from the two papers, and we suggest further extensions of context optimization for classification tasks with a fixed set of classes.

References:
- CoOp: https://arxiv.org/abs/2109.01134
- CoCoOp: https://arxiv.org/abs/2203.05557
