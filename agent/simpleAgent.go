package agent

import (
	"context"
	"llm_dev/model"

	"github.com/rs/zerolog/log"
)

type SimpleAgent struct {
	BaseAgent
}

func (agent *SimpleAgent) NewUserTask(sysprmpt string, model_name string, tool []model.ToolDef) {
	agentCtx := NewAgentContext("")
	agentCtx.model = model_name
	agentCtx.registerTool(tool)
	for {
		req := agentCtx.genRequest(replaceSysPrompt)
		stream, err := agent.model.CreateChatCompletionStream(context.TODO(), req)
		if err != nil {
			log.Error().Err(err).Msg("create chat completion stream failed")
			break
		}
		defer stream.Close()
		agent.handleResponse(stream, agentCtx)
		if agentCtx.done() {
			break
		}
	}
}
