package knowledgebase

import (
	"fmt"
	"llm_dev/agent"
	"llm_dev/utils"
	"testing"
)

func TestReplaceAgent(t *testing.T) {
	t.Run("test doc mgr", func(t *testing.T) {
		mgr := DocMgr{}
		res := mgr.loadFile("/root/workspace/agentic_rag/doc.json")
		mgr.process(res)
	})
}

func TestSummaryPrompt(t *testing.T) {
	t.Run("test doc mgr", func(t *testing.T) {
		mgr := DocMgr{}
		res := mgr.loadFile("/root/workspace/agentic_rag/doc.json")
		mgr.process(res)
		prompt, _ := mgr.summaryMethod()
		p := prompt(mgr.sectionMsp["## 1 Introduction"])
		fmt.Printf("%s\n", p)

		model := utils.NewModel("https://openrouter.ai/api/v1", "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")
		llmAgent := agent.SimpleAgent{
			BaseAgent: agent.NewBaseAgent("", *model),
		}
		llmAgent.NewUserTask(p, "anthropic/claude-haiku-4.5")
	})
}
