package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"llm_dev/model"
	"strings"

	"github.com/rs/zerolog/log"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

var semanticSearch = openai.FunctionDefinition{
	Name:   "semantic_search",
	Strict: true,
	Description: `
Use semantic search to match the text chunk that is similar to the query.
	`,
	Parameters: jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"query": {
				Type:        jsonschema.String,
				Description: "the query to search in the vector database",
			},
		},
		Required: []string{"query"},
	},
}
var searchBySeq = openai.FunctionDefinition{
	Name:   "search_by_seq",
	Strict: true,
	Description: `
Search text chunks by the sequence number.

When to use this tool:
- you get some text chunks and wnat to review the content near this text chunk, you can search by sequence number.
	`,
	Parameters: jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"seqs": {
				Type:        jsonschema.Array,
				Description: "the sequence number array to search",
				Items: &jsonschema.Definition{
					Type: jsonschema.Number,
				},
			},
		},
		Required: []string{"seqs"},
	},
}

var ragPrompt = `
You are a Retrieval-Augmented Generation (RAG) agent designed to understand and answer questions using search tools.

Your Job:
1. Use the avaliable tool to search the relevant content and understand the document.
2. Base your answers solely on retrieved content.
   - Do not hallucinate.
   - If no relevant content is found, explicitly state that the documents contain no matching information.
3. Answer the user's question with the retrieved content, cite and reference the content.

# Tool Usage

You have access to the following tools:
- semantic_search: use vector search to match the text that is semantic relevant to the query.
- search_by_seq: search by the text chunk sequence number, used when you want to get text content arround some text chunk

IMPORTANT: DO NOT output any of your reasoning or explaination, just use the tools and generate the final response.

`

type RagAgent struct {
	BaseAgent
	sysprompt       string
	buildTextVecMgr *BuildTextVectorMgr
	recordText      []TextChunk
}

func (agent *RagAgent) SetSysprompt() {
	var strBuilder strings.Builder
	strBuilder.WriteString(ragPrompt)
	strBuilder.WriteString("Answer the user question based on the following document.\n")
	strBuilder.WriteString("File name: output.md\n")
	strBuilder.WriteString("Document Table of Contents:\n")
	strBuilder.WriteString("```\n")
	strBuilder.WriteString(`
Llama 2: Open Foundation and Fine-Tuned Chat Models (line 1)
- GenAI, Meta (line 19)
- Abstract (line 22)

Contents (line 42)

1 Introduction (line 147)

2 Pretraining (line 259)
- 2.1 Pretraining Data (line 269)
- 2.2 Training Details (line 283)
  - 2.2.1 Training Hardware & Carbon Footprint (line 357)
- 2.3 Llama 2 Pretrained Model Evaluation (line 421)

3 Fine-tuning (line 543)
- 3.1 Supervised Fine-Tuning (SFT) (line 559)
- 3.2 Reinforcement Learning with Human Feedback (RLHF) (line 625)
  - 3.2.1 Human Preference Data Collection (line 640)
  - 3.2.2 Reward Modeling (line 693)
  - 3.2.3 Iterative Fine-Tuning (line 896)
- 3.3 System Message for Multi-Turn Consistency (line 1066)
- 3.4 RLHF Results (line 1149)
  - 3.4.1 Model-Based Evaluation (line 1152)
  - 3.4.2 Human Evaluation (line 1315)

4 Safety (line 1385)
- 4.1 Safety in Pretraining (line 1398)
- 4.2 Safety Fine-Tuning (line 1639)
  - 4.2.1 Safety Categories and Annotation Guidelines (line 1661)
  - 4.2.2 Safety Supervised Fine-Tuning (line 1689)
  - 4.2.3 Safety RLHF (line 1702)
  - 4.2.4 Context Distillation for Safety (line 1930)
- 4.3 Red Teaming (line 2032)
- 4.4 Safety Evaluation of Llama 2-Chat (line 2106)

5 Discussion (line 2241)
- 5.1 Learnings and Observations (line 2248)
- 5.2 Limitations and Ethical Considerations (line 2376)
- 5.3 Responsible Release Strategy (line 2420)

6 Related Work (line 2459)

7 Conclusion (line 2534)

References (line 2550)

A Appendix (line 3172)
- A.1 Contributions (line 3174)
  - A.1.1 Acknowledgments (line 3204)
- A.2 Additional Details for Pretraining (line 3257)
  - A.2.1 Architecture Changes Compared to Llama 1 (line 3260)
  - A.2.2 Additional Details for Pretrained Models Evaluation (line 3352)
- A.3 Additional Details for Fine-tuning (line 3689)
  - A.3.1 Detailed Statistics of Meta Human Preference Data (line 3696)
  - A.3.2 Curriculum Strategy for Meta Human Preference Data (line 3715)
  - A.3.3 Ablation on Ranking Loss with Preference Rating-based Margin for Reward Modeling (line 3725)
  - A.3.4 Ablation on Ranking Loss with Safety Auxiliary Loss for Reward Modeling (line 3805)
  - A.3.5 Additional Results for GAtt (line 3839)
  - A.3.6 How Far Can Model-Based Evaluation Go? (line 3891)
  - A.3.7 Human Evaluation (line 3953)
- A.4 Additional Details for Safety (line 4141)
  - A.4.1 Tension between Safety and Helpfulness in Reward Modeling (line 4144)
  - A.4.2 Qualitative Results on Safety Data Scaling (line 4156)
  - A.4.3 English Pronouns (line 4169)
  - A.4.4 Context Distillation Preprompts (line 4319)
  - A.4.5 Safety Errors: False Refusals and Vague Responses (line 4325)
  - A.4.6 Examples of Safety Evaluation (line 4844)
  - A.4.7 Description of Automatic Safety Benchmarks (line 4969)
  - A.4.8 Automatic Safety Benchmark Evaluation Results (line 5007)
- A.5 Data Annotation (line 5383)
  - A.5.1 SFT Annotation Instructions (line 5391)
  - A.5.2 Negative User Experience Categories (line 5428)
  - A.5.3 Quality Assurance Process (line 5654)
  - A.5.4 Annotator Selection (line 5654)
- A.6 Dataset Contamination (line 5690)
- A.7 Model Card (line 5861)
`)
	strBuilder.WriteString("```\n")
	agent.sysprompt = strBuilder.String()
	fmt.Printf("agent.sysprompt\n%s\n", agent.sysprompt)
}

func (agent *RagAgent) NewUserTask(userprompt string) {
	agentCtx := NewAgentContext(userprompt)
	agentCtx.registerTool(agent.GetToolDef())
	for {
		req := agentCtx.genRequest(systemPompt)
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

func (rag *RagAgent) GetToolDef() []model.ToolDef {
	semanticHandler := func(argsStr string) (string, error) {
		args := struct {
			Query string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return rag.semanticSearch(args.Query, nil), nil
	}
	searchSeqHandler := func(argsStr string) (string, error) {
		args := struct {
			Seqs []int
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return rag.searchBySeq(args.Seqs), nil
	}
	res := []model.ToolDef{
		{FunctionDefinition: semanticSearch, Handler: semanticHandler},
		{FunctionDefinition: searchBySeq, Handler: searchSeqHandler},
	}
	return res
}

func (rag *RagAgent) recordChunk(textSeq int) string {
	return ""
}
func (rag *RagAgent) searchBySeq(seqs []int) string {
	matchText := rag.buildTextVecMgr.querySeq(seqs)
	var resStr strings.Builder
	resStr.WriteString(`
Here are the text chunks matched by sequence number.
`)

	resStr.WriteString("```\n")
	for _, text := range matchText {
		resStr.WriteString(fmt.Sprintf("Sequence: %d\n", text.Seq))
		resStr.WriteString(fmt.Sprintf("Headings: %s\n", strings.Join(text.Headings, ",")))
		resStr.WriteString("<content>\n")
		resStr.WriteString(text.Content)
		resStr.WriteString("</content>\n")
	}
	resStr.WriteString("```\n")
	return resStr.String()
}
func (rag *RagAgent) semanticSearch(query string, headings []string) string {
	matchText := rag.buildTextVecMgr.semanticSearch(query, 10, headings...)
	var resStr strings.Builder
	resStr.WriteString(`
Here are the top-10 text chunks that matches the query sorted in descending order.
`)

	resStr.WriteString("```\n")
	for _, text := range matchText {
		resStr.WriteString(fmt.Sprintf("Sequence: %d\n", text.Seq))
		resStr.WriteString(fmt.Sprintf("Headings: %s\n", strings.Join(text.Headings, ",")))
		resStr.WriteString("<content>\n")
		resStr.WriteString(text.Content)
		resStr.WriteString("</content>\n")
	}
	resStr.WriteString("```\n")
	return resStr.String()
}
