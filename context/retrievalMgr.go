package context

import (
	"encoding/json"
	"fmt"
	"llm_dev/model"
	"llm_dev/utils"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/rs/zerolog/log"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

type Action struct {
	Argument string
	Result   []ChunkItem
}

func (a *Action) toString(builder *strings.Builder) {
	builder.WriteString(fmt.Sprintf("Action: %s\n", a.Argument))
	if len(a.Result) == 0 {
		builder.WriteString("NO Results\n")
	} else {
		builder.WriteString("Results:\n")
		for _, res := range a.Result {
			builder.WriteString(fmt.Sprintf("=== Result %d ===\n", res.id))
			builder.WriteString(res.Content())
			builder.WriteString("\n")
		}
	}
	builder.WriteString("\n")
}

type ChunkItem struct {
	id   int
	text string
	meta []byte
}

func (item *ChunkItem) Content() string {
	var metaData map[string]any
	err := json.Unmarshal(item.meta, &metaData)
	if err != nil {
		log.Error().Err(err).Msg("parsing chunk metadata failed")
		return item.text
	}
	summary, _ := metaData["summary"].(bool)
	if summary {
		heading, _ := metaData["heading"].(string)
		return fmt.Sprintf("Section ** %s ** summary:\n%s", heading, item.text)
	} else {
		return item.text
	}
}
func (item *ChunkItem) Metadata() string {
	return fmt.Sprintf("Metadata for result %d:\n%s\n", item.id, item.meta)
}

type Thought struct {
	Content string
	Status  string
}
type Observation struct {
	Statement string
	Ref       []ChunkItem
}

type RetrievalCtxMgr struct {
	db *utils.DBmgr

	CurrentThought *Thought
	Results        []*ChunkItem
	Actions        []Action
	Thoughts       []Thought
	Observations   []Observation
}

func (mgr *RetrievalCtxMgr) createThought(content string) string {
	if mgr.CurrentThought != nil {
		return fmt.Sprintf("Current thought: %s not finished, can not create new thought, please first finish the current thought", mgr.CurrentThought.Content)
	}
	mgr.CurrentThought = &Thought{
		Content: content,
		Status:  "In Progress",
	}
	return fmt.Sprintf("Create new thought: %s success", content)
}
func (mgr *RetrievalCtxMgr) finishThought() string {
	if mgr.CurrentThought == nil {
		return "There is no current thought, can not finish empty thought"
	}
	mgr.CurrentThought.Status = "Completed"
	mgr.Results = nil
	mgr.Actions = nil
	mgr.Thoughts = append(mgr.Thoughts, *mgr.CurrentThought)
	return fmt.Sprintf("Finish current thought: %s", mgr.CurrentThought.Content)
}
func (mgr *RetrievalCtxMgr) observe(statement string, refs []int) string {
	size := len(mgr.Results)
	res := []ChunkItem{}
	for _, id := range refs {
		if id < 0 || id >= size {
			return ""
		}
		res = append(res, *mgr.Results[id])
	}
	ob := Observation{
		Statement: statement,
		Ref:       res,
	}
	mgr.Observations = append(mgr.Observations, ob)
	return ""
}

func (mgr *RetrievalCtxMgr) query(heading string) []ChunkItem {
	filter := fmt.Sprintf(`heading == "%s"`, heading)
	var matches []ChunkItem
	err := mgr.db.Query(filter, []string{"text", "$meta"}, func(result *milvusclient.ResultSet) {
		textCol, ok1 := result.GetColumn("text").(*column.ColumnVarChar)
		metaCol, ok2 := result.GetColumn("$meta").(*column.ColumnJSONBytes)
		if !ok1 || !ok2 {
			log.Error().Any("text", ok1).Any("$meta", ok2).
				Msg("get column as concrete data type failed")
			return
		}
		textData := textCol.Data()
		metaData := metaCol.Data()
		size := textCol.Len()
		for i := range size {
			matches = append(matches, ChunkItem{
				text: textData[i],
				meta: metaData[i],
			})
		}
	})
	if err != nil {
		log.Error().Err(err).Msg("execute db query failed")
		return nil
	}
	return matches
}
func (mgr *RetrievalCtxMgr) searchSummary(text string) []ChunkItem {
	filter := "summary == true"
	var matches []ChunkItem
	err := mgr.db.Search(text, 5, filter, []string{"text", "$meta"}, func(results []milvusclient.ResultSet) {
		if len(results) != 1 {
			log.Error().Msg("rresult size is not 1")
			return
		}
		res := &results[0]
		textCol, ok1 := res.GetColumn("text").(*column.ColumnVarChar)
		metaCol, ok2 := res.GetColumn("$meta").(*column.ColumnJSONBytes)
		if !ok1 || !ok2 {
			log.Error().Any("text", ok1).Any("$meta", ok2).
				Msg("get column as concrete data type failed")
			return
		}
		textData := textCol.Data()
		metaData := metaCol.Data()
		size := textCol.Len()
		for i := range size {
			matches = append(matches, ChunkItem{
				text: textData[i],
				meta: metaData[i],
			})
		}
	})
	if err != nil {
		log.Error().Err(err).Msg("execute db Search failed")
		return nil
	}
	return matches
}

func (mgr *RetrievalCtxMgr) searchText(text string, heading string) []ChunkItem {
	filter := fmt.Sprintf(`ARRAY_CONTAINS(headings, "%s") AND summary IS NULL`, heading)
	// filter := "summary IS NULL"

	var matches []ChunkItem
	err := mgr.db.Search(text, 5, filter, []string{"text", "$meta"}, func(results []milvusclient.ResultSet) {
		if len(results) != 1 {
			log.Error().Msg("rresult size is not 1")
			return
		}
		res := &results[0]
		textCol, ok1 := res.GetColumn("text").(*column.ColumnVarChar)
		metaCol, ok2 := res.GetColumn("$meta").(*column.ColumnJSONBytes)
		if !ok1 || !ok2 {
			log.Error().Any("text", ok1).Any("$meta", ok2).
				Msg("get column as concrete data type failed")
			return
		}
		textData := textCol.Data()
		metaData := metaCol.Data()
		size := textCol.Len()
		for i := range size {
			matches = append(matches, ChunkItem{
				text: textData[i],
				meta: metaData[i],
			})
		}
	})
	if err != nil {
		log.Error().Err(err).Msg("execute db Search text failed")
		return nil
	}
	return matches
}
func (mgr *RetrievalCtxMgr) observeTool() model.ToolDef {
	var def = openai.FunctionDefinition{
		Name:   "record_observation",
		Strict: true,
		Description: `
Observe all the action result thoroughly, record all the factual statement derived from the action results that is helpful to the task.

Each observation record item must be:
- short and concise  
- contain a single fact  
- directly supported by the action result  
- include the source ID like (ref: chunk_12)  
- if the action result is useless, explict record that the action result is not what you are looking for
`,
		Parameters: jsonschema.Definition{
			Type:                 jsonschema.Object,
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"statement": {
					Type:        jsonschema.String,
					Description: "the single factual statement of the observation",
				},
				"refs": {
					Type:        jsonschema.Array,
					Description: "the sources id of this observation record",
					Items: &jsonschema.Definition{
						Type:        jsonschema.Number,
						Description: "the action result id",
					},
				},
			},
			Required: []string{"statement", "refs"},
		},
	}
	handler := func(argsStr string) (string, error) {
		args := struct {
			Statement string
			Refs      []int
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return mgr.observe(args.Statement, args.Refs), nil
	}
	return model.ToolDef{FunctionDefinition: def, Handler: handler}
}
func (mgr *RetrievalCtxMgr) thoughtTool() model.ToolDef {
	var def = openai.FunctionDefinition{
		Name:   "create_thought",
		Strict: true,
		Description: `
Resoning about the user task and all the context, decompose the task and think about next step, create thought.
`,
		Parameters: jsonschema.Definition{
			Type:                 jsonschema.Object,
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"content": {
					Type:        jsonschema.String,
					Description: "the content of the thought",
				},
			},
			Required: []string{"content"},
		},
	}
	handler := func(argsStr string) (string, error) {
		args := struct {
			Content string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return mgr.createThought(args.Content), nil
	}
	return model.ToolDef{FunctionDefinition: def, Handler: handler}
}
func (mgr *RetrievalCtxMgr) finishThoughtTool() model.ToolDef {
	var def = openai.FunctionDefinition{
		Name:   "finish_thought",
		Strict: true,
		Description: `
Finish current in progress thought.
`,
		Parameters: jsonschema.Definition{
			Type:                 jsonschema.Object,
			AdditionalProperties: false,
			Properties:           map[string]jsonschema.Definition{},
			Required:             []string{},
		},
	}
	handler := func(argsStr string) (string, error) {
		return mgr.finishThought(), nil
	}
	return model.ToolDef{FunctionDefinition: def, Handler: handler}
}
func (mgr *RetrievalCtxMgr) metadataTool() model.ToolDef {
	var def = openai.FunctionDefinition{
		Name:   "get_metadata",
		Strict: true,
		Description: `
Get the metadata of the action result.
For a document chunk result, the metadata include the chunk sequence and the section headings of the chunk.
For a section summary, the metadata include the parent section heading and sub section headings.
`,
		Parameters: jsonschema.Definition{
			Type:                 jsonschema.Object,
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"id": {
					Type:        jsonschema.Number,
					Description: "the id of the action result",
				},
			},
			Required: []string{"id"},
		},
	}
	handler := func(argsStr string) (string, error) {
		args := struct {
			Id int
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		if args.Id < 0 || args.Id >= len(mgr.Results) {
			return fmt.Sprintf("id %d out of bound", args.Id), nil
		}
		return mgr.Results[args.Id].Metadata(), nil
	}
	return model.ToolDef{FunctionDefinition: def, Handler: handler}
}
func (mgr *RetrievalCtxMgr) queryTool() model.ToolDef {
	var def = openai.FunctionDefinition{
		Name:   "get_summary_by_heading",
		Strict: true,
		Description: `
Search section summary by its heading.
`,
		Parameters: jsonschema.Definition{
			Type:                 jsonschema.Object,
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"heading": {
					Type:        jsonschema.String,
					Description: "the heading of the section, e.g. '## introduction', '# section 1', '## contents'",
				},
			},
			Required: []string{"heading"},
		},
	}
	handler := func(argsStr string) (string, error) {
		args := struct {
			Heading string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		action := Action{
			Argument: fmt.Sprintf(""),
			Result:   mgr.query(args.Heading),
		}
		mgr.Actions = append(mgr.Actions, action)
		return "", nil
	}
	return model.ToolDef{FunctionDefinition: def, Handler: handler}
}
func (mgr *RetrievalCtxMgr) searchSummaryTool() model.ToolDef {
	var def = openai.FunctionDefinition{
		Name:   "vector_search_summary",
		Strict: true,
		Description: `
The vector database stores the section summary infomation, you can use vector search to search the section summary using a query.
`,
		Parameters: jsonschema.Definition{
			Type:                 jsonschema.Object,
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"query": {
					Type:        jsonschema.String,
					Description: "the query for vector search",
				},
			},
			Required: []string{"query"},
		},
	}
	handler := func(argsStr string) (string, error) {
		args := struct {
			Query string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		action := Action{
			Argument: fmt.Sprintf("vector search summary with query: %s", args.Query),
			Result:   mgr.searchSummary(args.Query),
		}
		mgr.Actions = append(mgr.Actions, action)
		return "run vector search success", nil
	}
	return model.ToolDef{FunctionDefinition: def, Handler: handler}
}
func (mgr *RetrievalCtxMgr) searchTextTool() model.ToolDef {
	var def = openai.FunctionDefinition{
		Name:   "vector_search_text",
		Strict: true,
		Description: `
The vector database stores document chunks, you can use vector search to search the document chunks.

Usage:
- specify the query to do vector search
- specify the heading to narrow the search scope, the vector search will search all the chunks under that heading section.
`,
		Parameters: jsonschema.Definition{
			Type:                 jsonschema.Object,
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"query": {
					Type:        jsonschema.String,
					Description: "the query for vector search",
				},
				"heading": {
					Type:        jsonschema.String,
					Description: "the section heading which is the vector search scope",
				},
			},
			Required: []string{"query", "heading"},
		},
	}
	handler := func(argsStr string) (string, error) {
		args := struct {
			Query   string
			Heading string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		action := Action{
			Argument: fmt.Sprintf("vector search doc chunks with query: %s", args.Query),
			Result:   mgr.searchText(args.Query, args.Heading),
		}
		mgr.Actions = append(mgr.Actions, action)
		return "run vector search success", nil
	}
	return model.ToolDef{FunctionDefinition: def, Handler: handler}
}

func (mgr *RetrievalCtxMgr) genSysPrompt() string {
	var builder strings.Builder
	instruct := `
You are an agentic RAG system using the ReAct (Reason + Act) paradigm.

### WORKFLOW ###

1. Use "Thought:" to decide what to do.
2. Use "Action:" to call tools. Only use tools when needed.
3. After the tool executes, read the tool output and observe, record the conclusion of the observation.
4. Repeat Thought → Action → Observation until enough information is gathered.
5. Produce "Final Answer:" at the end using ONLY retrieved information.
6. Never include internal reasoning outside of "Thought:" tags.

### RULES ###

** Thought ** rules:
- MUST finish current thought before you create next thought.
- Perform multi-step decomposition for complex questions.
- Prefer multiple sub-queries over one large query.

** Observation ** rules:
1. While excute actions, you should IMMEDIATELY record the factual statement derived from the action results that is helpful for the task.
2. Observe all the action results and record A bullet list of factual statements derived from the action results that is helpful to the task.
Each item must be:
- short and concise  
- contain a single fact  
- directly supported by the action result  
- include the source ID like (ref: chunk_12)  
3. if the action result is useless, explict record that the action result is not what you want

<example>
- CRISPR allows precise gene editing (ref: chunk_4)
- Viral vectors deliver DNA into cells (ref: chunk_18)
</example>

** Final Answer ** rules:
- Never hallucinate facts.
- Only answer using retrieved content.
- Do not answer until retrieval is complete.

### TOOL GUIDELINE ###

You have access to the following tools:
- create_thought: think about what to do next and create thought.
- finish_thought: finish the current thought
- record_observation: record all the factual statement derived from action result.
- get_metadata: get metadata of some action result.
- get_summary_by_heading: search section summary by its heading
- vector_search_summary: using vector search to find relevant content from the seciton summary.
- vector_search_text: using vector search to find relevant document chunks within some scope.

`

	for i := range mgr.Actions {
		for j := range mgr.Actions[i].Result {
			mgr.Results = append(mgr.Results, &mgr.Actions[i].Result[j])
		}
	}
	for i, res := range mgr.Results {
		res.id = i
	}
	builder.WriteString(instruct)
	builder.WriteString("### CONTEXT ###\n")
	if len(mgr.Thoughts) != 0 {
		builder.WriteString("# ** Previous Thoughts **\n\n")
		for _, t := range mgr.Thoughts {
			fmt.Fprintf(&builder, "- %s (%s)\n", t.Content, t.Status)
		}
		builder.WriteString("\n")
	}
	if len(mgr.Observations) != 0 {
		builder.WriteString("# ** Observations **\n\n")
		for _, o := range mgr.Observations {
			refs := strings.Trim(fmt.Sprint(o.Ref), "[]")
			fmt.Fprintf(&builder, "- %s (Refs: %s)", o.Statement, refs)
		}
	}
	builder.WriteString("# ** In progress Thought and Actions **\n\n")
	if mgr.CurrentThought == nil {
		builder.WriteString("There is no current thought\n")
	} else {
		fmt.Fprintf(&builder, "Current thought: %s\n\n", mgr.CurrentThought.Content)
		builder.WriteString("** Action & Result **\n")
		for _, a := range mgr.Actions {
			a.toString(&builder)
		}
	}
	return builder.String()
}
