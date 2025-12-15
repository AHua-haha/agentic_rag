package agent

import (
	"context"
	"fmt"
	ctx "llm_dev/context"
	"llm_dev/utils"
	"os"

	"github.com/rs/zerolog/log"
)

type RetrievalAgent struct {
	BaseAgent
}

func (ragAgent *RetrievalAgent) retrieval(query string) {

	db, err := utils.NewDBMgr()
	if err != nil {
		log.Error().Err(err).Msg("create db mgr failed")
		return
	}
	defer db.Close()
	retrievalMgr := ctx.RetrievalCtxMgr{
		DB:        db,
		UserQuery: query,
	}
	agentCtx := NewAgentContext(query)
	agentCtx.model = "anthropic/claude-sonnet-4.5"
	agentCtx.registerTool(retrievalMgr.Tools())

	for {
		ctxStr := retrievalMgr.GenSysPrompt()
		file, _ := os.OpenFile("context.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		file.WriteString("Prompt:\n")
		file.WriteString(ctxStr)
		file.Close()
		req := agentCtx.genRequest(ctxStr)
		stream, err := ragAgent.model.CreateChatCompletionStream(context.TODO(), req)
		if err != nil {
			file, _ := os.OpenFile("context.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			for i := range req.Messages {
				fmt.Fprintf(file, "%#v\n", req.Messages[i])
			}
			file.Close()
			log.Error().Err(err).Msg("create chat completion stream failed")
			break
		}
		defer stream.Close()
		ragAgent.handleResponse(stream, agentCtx)
		if agentCtx.done() {
			break
		}
	}
}
