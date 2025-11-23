package agent

import (
	"testing"
)

func TestReplaceAgent(t *testing.T) {
	t.Run("test replace agent", func(t *testing.T) {
		model := NewModel("http://192.168.65.2:4000", "sk-1234")
		repAgent := ReplaceAgent{
			BaseAgent: NewBaseAgent("", *model),
		}
		toc := `
Llama 2: Open Foundation and Fine-Tuned Chat Models (line 1)
- GenAI, Meta (line 19)
- Abstract (line 22)

Contents (line 42)

1 Introduction (line 147)

2 Pretraining (line 259)
- 2.1 Pretraining Data (line 269)
- 2.2 Training Details (line 283)
  - 2.2.1 Training Hardware & Carbon Footprint (line 357)
- 2.3 Llama 2 Pretrained Model Evaluation (line 421)

3 Fine-tuning (line 543)
- 3.1 Supervised Fine-Tuning (SFT) (line 559)
- 3.2 Reinforcement Learning with Human Feedback (RLHF) (line 625)
  - 3.2.1 Human Preference Data Collection (line 640)
  - 3.2.2 Reward Modeling (line 693)
  - 3.2.3 Iterative Fine-Tuning (line 896)
- 3.3 System Message for Multi-Turn Consistency (line 1066)
- 3.4 RLHF Results (line 1149)
  - 3.4.1 Model-Based Evaluation (line 1152)
  - 3.4.2 Human Evaluation (line 1315)

4 Safety (line 1385)
- 4.1 Safety in Pretraining (line 1398)
- 4.2 Safety Fine-Tuning (line 1639)
  - 4.2.1 Safety Categories and Annotation Guidelines (line 1661)
  - 4.2.2 Safety Supervised Fine-Tuning (line 1689)
  - 4.2.3 Safety RLHF (line 1702)
  - 4.2.4 Context Distillation for Safety (line 1930)
- 4.3 Red Teaming (line 2032)
- 4.4 Safety Evaluation of Llama 2-Chat (line 2106)

5 Discussion (line 2241)
- 5.1 Learnings and Observations (line 2248)
- 5.2 Limitations and Ethical Considerations (line 2376)
- 5.3 Responsible Release Strategy (line 2420)

6 Related Work (line 2459)

7 Conclusion (line 2534)

References (line 2550)

A Appendix (line 3172)
- A.1 Contributions (line 3174)
  - A.1.1 Acknowledgments (line 3204)
- A.2 Additional Details for Pretraining (line 3257)
  - A.2.1 Architecture Changes Compared to Llama 1 (line 3260)
  - A.2.2 Additional Details for Pretrained Models Evaluation (line 3352)
- A.3 Additional Details for Fine-tuning (line 3689)
  - A.3.1 Detailed Statistics of Meta Human Preference Data (line 3696)
  - A.3.2 Curriculum Strategy for Meta Human Preference Data (line 3715)
  - A.3.3 Ablation on Ranking Loss with Preference Rating-based Margin for Reward Modeling (line 3725)
  - A.3.4 Ablation on Ranking Loss with Safety Auxiliary Loss for Reward Modeling (line 3805)
  - A.3.5 Additional Results for GAtt (line 3839)
  - A.3.6 How Far Can Model-Based Evaluation Go? (line 3891)
  - A.3.7 Human Evaluation (line 3953)
- A.4 Additional Details for Safety (line 4141)
  - A.4.1 Tension between Safety and Helpfulness in Reward Modeling (line 4144)
  - A.4.2 Qualitative Results on Safety Data Scaling (line 4156)
  - A.4.3 English Pronouns (line 4169)
  - A.4.4 Context Distillation Preprompts (line 4319)
  - A.4.5 Safety Errors: False Refusals and Vague Responses (line 4325)
  - A.4.6 Examples of Safety Evaluation (line 4844)
  - A.4.7 Description of Automatic Safety Benchmarks (line 4969)
  - A.4.8 Automatic Safety Benchmark Evaluation Results (line 5007)
- A.5 Data Annotation (line 5383)
  - A.5.1 SFT Annotation Instructions (line 5391)
  - A.5.2 Negative User Experience Categories (line 5428)
  - A.5.3 Quality Assurance Process (line 5654)
  - A.5.4 Annotator Selection (line 5654)
- A.6 Dataset Contamination (line 5690)
- A.7 Model Card (line 5861)
`
		repAgent.NewUserTask("/root/workspace/agentic_rag/output.md", toc)
	})
}
