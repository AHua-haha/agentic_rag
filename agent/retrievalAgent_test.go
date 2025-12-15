package agent

import (
	"fmt"
	"llm_dev/utils"
	"testing"
)

func TestRetrievalAgent_retrieval(t *testing.T) {
	t.Run("test rag", func(t *testing.T) {
		// TODO: construct the receiver type.
		model := utils.NewModel("https://openrouter.ai/api/v1", "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")
		agent := RetrievalAgent{
			BaseAgent: NewBaseAgent("", *model),
		}
		agent.retrieval("Discuss the significance of the \"Reinforcement Learning with Human Feedback (RLHF)\" method described in section 3.2 of the \"llama2.pdf\" document. How does it differ from the Supervised Fine-Tuning approach mentioned in section 3.1?")
	})
}

func TestEmbedding(t *testing.T) {
	t.Run("test embedding", func(t *testing.T) {
		res, err := utils.EmbedText([]string{
			"hello world",
			"da da ta da",
		})
		if err != nil {
			fmt.Printf("err: %v\n", err)
			return
		}
		fmt.Printf("len(res): %v\n", len(res))
	})
}
