package agent

import (
	"llm_dev/utils"
	"testing"
)

func TestSummary(t *testing.T) {
	t.Run("test summary agent", func(t *testing.T) {
		// TODO: construct the receiver type.
		model := utils.NewModel("http://172.17.0.1:4000", "sk-1234")
		agent := SummarizeAgent{
			BaseAgent: NewBaseAgent("", *model),
		}
		agent.genToc("openrouter/anthropic/claude-sonnet-4.5", "/root/workspace/agentic_rag/MinerU_2307.09288v2__20251127030211.md")
	})
}
