package agent

import (
	"context"
	ctx "llm_dev/context"
	"llm_dev/utils"
	"os"

	"github.com/rs/zerolog/log"
)

type SummarizeAgent struct {
	BaseAgent
}

func (agent *SummarizeAgent) genSummary(model_name string, file string) {
	mgr := ctx.NewChunkCtxMgr(file)
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
		agentCtx := NewAgentContext(user)
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
	db, err := utils.NewDBMgr()
	if err != nil {
		log.Error().Err(err).Msg("create db mgr failed")
		return
	}
	defer db.Close()
	cols, err := mgr.GenCols()
	if err != nil {
		log.Error().Err(err).Msg("generate cols for parts summary failed")
		return
	}
	err = db.Insert(cols)
	if err != nil {
		log.Error().Err(err).Msg("insert columns into db failed")
		return
	}
	log.Info().Msg("generate summary and insert into db success")
}
