package agent

import (
	"context"
	ctx "llm_dev/context"

	"github.com/rs/zerolog/log"
)

var ragPrompt = `
You are a Retrieval-Augmented Generation (RAG) agent designed to understand and answer questions.
You have the access to a knowledge base, you can use tools to search for relevant context in the knowledge base.

Your Job:
1. Use the avaliable tool to search the relevant content and understand the document.
2. Base your answers solely on retrieved content.
   - Do not hallucinate.
   - If no relevant content is found, explicitly state that the documents contain no matching information.
3. Answer the user's question with the retrieved content, cite and reference the content.

IMPORTANT: DO NOT output any of your reasoning or explaination, just use the tools and generate the final response.
- 

`

type RagAgent struct {
	BaseAgent
}

func (agent *RagAgent) NewUserTask(userprompt string) {
	knowledgebaseMgr := ctx.NewKnowledgeBase("output.md")
	agentCtx := NewAgentContext(userprompt, &knowledgebaseMgr)
	for {
		req := agentCtx.genRequest(ragPrompt)
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
