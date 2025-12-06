package agent

import (
	"context"
	ctx "llm_dev/context"
	"os"

	"github.com/rs/zerolog/log"
)

type SummarizeAgent struct {
	BaseAgent
}

func (agent *SummarizeAgent) genToc(model_name string, file string) {
	mgr := ctx.NewSummaryMgr(file)
	for {
		sys, user, tool, finished := mgr.Next()
		if finished {
			break
		}
		file, _ := os.OpenFile("context.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		file.WriteString("User:\n")
		file.WriteString(user)
		file.WriteString("\n")
		file.Close()
		agentCtx := NewAgentContext(user, &mgr)
		agentCtx.model = model_name
		agentCtx.registerTool(tool)
		var e error
		for {
			req := agentCtx.genRequest(sys)
			stream, err := agent.model.CreateChatCompletionStream(context.TODO(), req)
			if err != nil {
				e = err
				log.Error().Err(err).Msg("create chat completion stream failed")
				break
			}
			defer stream.Close()
			agent.handleResponse(stream, agentCtx)
			if agentCtx.done() {
				break
			}
		}
		if e != nil {
			break
		}
	}
}
