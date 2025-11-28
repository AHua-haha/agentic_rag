package agent

import (
	"llm_dev/utils"
	_ "llm_dev/utils"
	"testing"
)

func TestChunkAgent(t *testing.T) {
	t.Run("test agentic chunk", func(t *testing.T) {
		// TODO: construct the receiver type.
		model := utils.NewModel("http://192.168.65.2:4000", "sk-1234")
		chunkagent := ChunkAgent{
			BaseAgent: NewBaseAgent("", *model),
		}
		chunkagent.NewUserTask("/root/workspace/agentic_rag/output.md")
	})
}
