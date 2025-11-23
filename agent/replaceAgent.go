package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"llm_dev/model"
	"os/exec"
	"strings"

	"github.com/rs/zerolog/log"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

var replaceSysPrompt = `
You are a dedicated Document Refinement Agent. 
Your sole responsibility is to format the document headings based on a previous generated table of content of the document.

Requirements:
1. you MUST use the 'replace' tool which is build on command tool 'sed' to do in place replace, each item in table of content have a line number, replace based on line number.
2. the document is markdown format, you MUST format each heading in markdown format, use '#' for heading.
3. you MUST ensure the correct heading hierarchy consistency with the table of contents, that is to say ensure each markdown heading level is correct.

`

var replaceTool = openai.FunctionDefinition{
	Name:   "replace",
	Strict: true,
	Description: `
Text replace tool build on command line tool 'sed'.
	`,
	Parameters: jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"arguments": {
				Type:        jsonschema.String,
				Description: "the argument to pass to sed",
			},
		},
		Required: []string{"arguments"},
	},
}

func replaceToolDef() model.ToolDef {
	replaceHandler := func(argsStr string) (string, error) {
		args := struct {
			Arguments string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		sedStr := fmt.Sprintf("sed %s", args.Arguments)
		cmd := exec.Command("bash", "-c", sedStr)
		output, err := cmd.Output()
		if err != nil {
			return fmt.Sprintf("Execute sed failed, Error:%s, output: %s", err, output), nil
		}
		return "Execute sed success", nil
	}

	return model.ToolDef{
		FunctionDefinition: replaceTool,
		Handler:            replaceHandler,
	}
}

type ReplaceAgent struct {
	BaseAgent
}

func (agent *ReplaceAgent) NewUserTask(filepath string, toc string) {
	var strBuilder strings.Builder
	strBuilder.WriteString("Format the document headings based on the table of contents.\n")
	strBuilder.WriteString(fmt.Sprintf("file: %s\n", filepath))
	strBuilder.WriteString("Table of Contents:\n")
	strBuilder.WriteString("```\n")
	strBuilder.WriteString(toc)
	strBuilder.WriteString("```\n")
	agentCtx := NewAgentContext(strBuilder.String())
	agentCtx.registerTool([]model.ToolDef{replaceToolDef()})
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
