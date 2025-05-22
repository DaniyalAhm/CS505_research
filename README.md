# Can an Ensemble of Distilled Models Mimic the Behavior of a High Parameter LLM

**Daniyal Ahmed**
Boston University
[daniyal@bu.edu](mailto:daniyal@bu.edu)

---

## Abstract

To help mitigate environmental costs of Large Language Models (LLMs), I explore if 'stacking' fine-tuned distilled models can achieve similar performance to a high-parameter 'teacher' model.

---

## 1. Introduction

In this project, I originally set out to answer the following question: **Can an ensemble of distilled language models mimic a high-parameter 'teacher' model?** This idea was inspired by Mixture of Experts, but instead of using gating mechanisms to assign experts, I employ ensembles of specialized distilled models.

---

## 2. Goal

The goal of this research project was to determine whether fine-tuning several distilled models on specific tasks can improve their performance in those domains. By combining these specialist models into an ensemble, we aim to approximate the behavior of a larger, more generalist model while using significantly fewer resources.

---

## 3. Method

1. **Select distilled models:** e.g., Deepseek-R1-Distilled-Qwen-1.5B.
2. **Fine-tune specialists** on targeted datasets (math, coding, etc.).
3. **Train a delegator** (using an encoding model like BERT) to classify incoming queries and route them to the appropriate specialist.
4. **Ensemble response:** The specialist generates an answer, which is returned to the user.

A typical interaction:

* User: "Write me some code."
* Delegator: identifies the intent as coding and forwards the request to the coding specialist.
* Specialist: generates code and sends it back.

---

## 4. Hypothesis

Because each specialist is fine-tuned on a specific topic, it should provide more nuanced, higher-quality answers in its domain than a generalized model. Therefore, the ensemble's overall evaluation scores should match or exceed those of the teacher model.

---

## 5. Training the Math Specialist (First Attempt)

* **Dataset:** OpenR1 Math (220K samples)
* **Setup:**

  * 4-bit LoRA quantization
  * Max tokens: 512
  * Samples: 20,000
  * Learning rate: 2e-5, Adam8bit
  * Epochs: 1 (to prevent overfitting)

**Result:** Accuracy dropped to zero, indicating that the model failed to learn the desired patterns.

---

## 6. Employing More Training Data

* Increased max tokens to 1024
* Epochs: 3

**Result:** No performance improvement; accuracy remained at zero.

---

## 7. Switching to 8-bit Quantization

* Samples increased to 20,000
* 8-bit quantization

**Result:** Still failed to learn basic tagging patterns (e.g., `<Answer/>`) consistently.

---

## 8. Evaluation

| Model                                      | AIME pass\@1 | Math-500 pass\@1 |
| ------------------------------------------ | ------------ | ---------------- |
| Deepseek-R1-Distilled-Qwen-1.5B (Baseline) | 28.9         | 83.9             |
| Finetuned Deepseek-R1-Distilled-Qwen-1.5B  | 0            | 3.8              |

*Performance vs. baseline decreased dramatically.*

---

## 9. Training the Coding Specialist

* **Dataset:** CodeParrot/apps (10,000 samples, 1024 tokens)

| Model                                      | LiveBench pass\@1 |
| ------------------------------------------ | ----------------- |
| Deepseek-R1-Distilled-Qwen-1.5B (Baseline) | 16.9              |
| Finetuned Deepseek-R1-Distilled-Qwen-1.5B  | 0                 |

*The model could not follow code tagging, making evaluation infeasible.*

---

## 10. Catastrophic Forgetting

Repeated fine-tuning led to further degradation. This aligns with findings by Luo *et al.* (2025) on catastrophic forgetting during continual fine-tuning. Distilled models, being compressed, may be more susceptible to forgetting basic patterns.

---

## 11. Testing for Catastrophic Forgetting

Reducing training samples (20K → 10K) yielded occasional correct tagging, suggesting overfitting exacerbates forgetting.

---

## 12. The Delegator

Planned to fine-tune a delegator alongside specialists, but abandoned due to specialist failures. Delegator methods have been studied extensively (F. Roma *et al.*, 2023).

---

## 13. Replicability

All code is publicly available:
[GitHub Repository](https://github.com/DaniyalAhm/CS505_research)

---

## 14. Conclusion

Distilled models suffer severe catastrophic forgetting when fine-tuned on specific tasks. Their compressed nature limits how much new knowledge they can acquire before basic patterns are lost. Further research is needed to mitigate forgetting in specialist fine-tuning.

---

## References

1. Open-R1/OpenR1-Math-220K · Datasets at Hugging Face (Accessed: 02 March 2025)
2. HuggingFaceH4/MATH-500 · Datasets at Hugging Face (Accessed: 02 February 2025)
3. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv* (1503.02531).
4. LiveBench/coding · Datasets at Hugging Face (Accessed: 02 February 2025)
5. Luo, Y., *et al.* (2025). *An empirical study of catastrophic forgetting in large language models during continual fine-tuning*. arXiv.
6. Roma, F., Sansonetti, G., D’Aniello, G., & Micarelli, A. (2023). *A BERT-Based Approach to Intent Recognition*. IEEE EUROCON.
7. McLean, S. (2024). The environmental impact of ChatGPT. Earth.Org.
8. Guo, D., *et al.* (2025). *Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning*. arXiv.
9. Other references as listed in the original document.
