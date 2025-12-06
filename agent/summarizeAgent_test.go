package agent

import (
	"llm_dev/utils"
	"testing"
)

func TestSummary(t *testing.T) {
	t.Run("test summary agent", func(t *testing.T) {
		// TODO: construct the receiver type.

		model := utils.NewModel("https://openrouter.ai/api/v1", "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")
		agent := SummarizeAgent{
			BaseAgent: NewBaseAgent("", *model),
		}
		agent.genToc("openai/gpt-5-mini", "/root/workspace/agentic_rag/MinerU_2307.09288v2__20251127030211.md")
	})
}
