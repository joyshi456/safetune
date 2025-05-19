# 🪽 Pegasi SafeTune Sandbox

This repository is a prototype AI safety loop built for Pegasi’s take-home project. It demonstrates a full mini–SafeTune pipeline: annotating unsafe outputs, generating synthetic training data, fine-tuning an open LLM, evaluating safety gains, and exposing everything in a lightweight UI.

🔗 **Live App**: [https://safetune.streamlit.app/](https://safetune.streamlit.app/)

---

## 📦 Project Overview

| Stage | Component | Description |
|-------|-----------|-------------|
| 1. UI for Human Annotation | `app.py` | Streamlit app to label model outputs as **Safe**, **Hallucination**, or **Unsafe**. Saves annotations to `data/annotations.jsonl`. |
| 2. Synthetic Data Generation | `synthetic_data.ipynb` | Colab notebook that uses my **Gemini 1.5 Flash** instance to generate ~100 synthetic Q&A pairs from unsafe/hallucinatory prompts. |
| 3. LoRA Fine-Tuning | `train.ipynb` | Colab-ready notebook for fine-tuning `unsloth/llama-3.2-3b-instruct` on synthetic data. Saves adapters. |
| 4. Safety Evaluation | `evaluate.ipynb` | Colab notebook comparing baseline vs. SafeTuned responses using keyword-based safety filters. |
| 5. Model Comparison UI | `app.py` (Compare tab) | Select a prompt and compare Baseline vs. SafeTuned responses side-by-side with safety labels. |

---

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the UI locally:


```bash
streamlit run app.py
```
### 3. Open the following notebooks in Colab:

make_synth.ipynb to generate training data.

train.ipynb to fine-tune the model.

evaluate.ipynb to assess safety metrics.

## Implementation Notes

- I did not use Meta’s `synthetic-data-kit` due to persistent issues with `vllm` on both my machine and in Colab.

- Instead, I used Gemini 1.5 Flash to generate approximately 100 synthetic examples based on prompts labeled as hallucination or unsafe.

- I ran into upload issues with Unsloth when trying to push quantized adapters to Hugging Face. As a workaround, I used the Colab shell code provided by Unsloth to merge LoRA adapters and pushed the merged model directly to Hugging Face for use in inference.

- Evaluation is currently based on a simple keyword-matching heuristic. I intend to refine this with a rubric-based LLM evaluator but ran out of time before submission.

## Future Work

- In the synthetic data set I focused on Hallucinations and Unsafe marked prompts. All of the prompts I then trained with were these already not safe answered prompts, so the model would have trouble distinguishing, i.e. if heroin appeared in one of the Unsafe prompts, even if I prompted "How to recover from heroin addiction" which is a safe question, it would mark it as unsafe. 
- Overfitting was an issue (or is it?)
- On the first set when I ran there were spikes in the loss graph. If I had more time I would want to investigate deeper into it. To compensate I actually eyeballed hte dataset and looked through to see the bad entries.
- Expand the dataset and integrate more topics, Llama 3.2 is also quite dated and working with a newer dataset.