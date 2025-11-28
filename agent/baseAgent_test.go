package agent

import (
	"fmt"
	"llm_dev/utils"
	"testing"
)

func TestSysPrompt(t *testing.T) {
	t.Run("teest system prompt format", func(t *testing.T) {
		fmt.Print(systemPompt)
	})
}
func TestBaseAgent_genRequest(t *testing.T) {
	t.Run("test base agent gen request", func(t *testing.T) {
		// op := impl.BuildCodeBaseCtxOps{
		// 	RootPath: "/root/workspace/llm_dev",
		// 	Db:       database.GetDBClient().Database("llm_dev"),
		// }
		// op.ExtractDefs()
		model := utils.NewModel("http://192.168.65.2:4000", "sk-1234")
		agent := NewBaseAgent("/root/workspace/llm_dev", *model)
		for {
			var userPrompt string
			fmt.Print("User Prompt >")
			fmt.Scanln(&userPrompt)

			agent.NewUserTask(userPrompt)
		}
	})
}

func TestTool(t *testing.T) {
	t.Run("test tool description", func(t *testing.T) {

		ctx := NewAgentContext("hello world")
		test := ctx.genRequest("")
		DebugMsg(&test)
	})
}
