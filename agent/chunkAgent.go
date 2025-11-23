package agent

import (
	"context"
	"fmt"
	ctx "llm_dev/context"

	"github.com/rs/zerolog/log"
)

var chunkPrompt = `
You are a specialized Document Structure Analyst agent. Your primary function is to meticulously analyze the provided text content, 
identify its hierarchical structure, and generate an accurate, nested Table of Contents (TOC).

IMPORTANT: The document is markdown format, but it may be converted from a pdf file, so the content may not be very well formated. you must do A thorough and comprehensive examination and analysis
IMPORTANT: DO NOT output any of your reasoning or explaination, just use the tools and generate the final result.
`

type ChunkAgent struct {
	BaseAgent
}

func (agent *ChunkAgent) NewUserTask(filepath string) {
	chunkCtxMgr := ctx.AgenticChunkCtxMgr{
		FilePath: filepath,
	}
	userprompt := fmt.Sprintf("Analyze the document and generate the table of content for file %s", filepath)
	agentCtx := NewAgentContext(userprompt, &chunkCtxMgr)
	for {
		req := agentCtx.genRequest(chunkPrompt)
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
