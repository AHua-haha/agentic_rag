package context

import (
	"encoding/json"
	"fmt"
	"llm_dev/model"
	"llm_dev/utils"
	"os"
	"regexp"
	"strings"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/rs/zerolog/log"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

type Part struct {
	Heading string
	Level   int
	Content string
	Child   []*Part
	Next    int

	Headings []string

	Summary    string
	Summarized bool
}

func (p *Part) canSummary() bool {
	for _, c := range p.Child {
		if !c.Summarized {
			return false
		}
	}
	return true
}

type FileChunkOps struct {
	File  string
	parts []Part
}

func (op *FileChunkOps) foldSection(window int) []Part {
	count := 0
	i := 0
	for i < len(op.parts) {
		if op.parts[i].Summarized {
			i = op.parts[i].Next
		} else {
			count++
			i++
		}
		if count >= window {
			break
		}
	}
	return op.parts[:i]
}

func (op *FileChunkOps) summary(id int, content string) string {
	if id < 0 || id >= len(op.parts) {
		return fmt.Sprintf("id %d out of range", id)
	}
	if op.parts[id].Summarized {
		return fmt.Sprintf("id %d is already summarized", id)
	}
	op.parts[id].Summarized = true
	op.parts[id].Summary = content
	return fmt.Sprintf("Summarized %d success", id)
}
func (op *FileChunkOps) buildTree() {
	size := len(op.parts)
	for i, _ := range op.parts {
		part := &op.parts[i]
		j := i + 1
		for ; j < size; j++ {
			if op.parts[j].Level <= part.Level {
				break
			}
			if op.parts[j].Level == part.Level+1 {
				part.Child = append(part.Child, &op.parts[j])
			}
		}
		part.Next = j
	}
	headings := make([]string, 6)
	for i := range op.parts {
		part := &op.parts[i]
		headings[part.Level-1] = part.Heading
		part.Headings = append(part.Headings, headings[:part.Level]...)
	}
}

func (op *FileChunkOps) chunk() {
	data, err := os.ReadFile(op.File)
	if err != nil {
		log.Error().Err(err).Msg("read file failed")
		return
	}
	re := regexp.MustCompile(`(?m)^(#{1,6})[ \t]+(.+?)\s*$`)
	matches := re.FindAllStringSubmatchIndex(string(data), -1)
	var sections []Part
	var idx []int

	for _, m := range matches {
		heading_s, heading_e := m[0], m[1]
		idx = append(idx, heading_s, heading_e)
	}
	idx = append(idx, len(data))

	if idx[0] != 0 {
		sections = append(sections, Part{
			Content: string(data[0:idx[0]]),
		})
	}

	for i := 0; i < len(idx)-1; i += 2 {
		h_s := idx[i]
		h_e := idx[i+1]
		end := idx[i+2]
		level := 0
		hash := strings.TrimSpace(string(data[h_s:h_e]))
		for _, c := range hash {
			if c == '#' {
				level++
			} else {
				break
			}
		}
		sections = append(sections, Part{
			Heading: strings.TrimSpace(string(data[h_s:h_e])),
			Level:   level,
			Content: string(data[h_e:end]),
		})
	}
	op.parts = sections
}

type ChunkCtxMgr struct {
	op FileChunkOps
}

func NewChunkCtxMgr(file string) ChunkCtxMgr {
	op := FileChunkOps{
		File: file,
	}
	op.chunk()
	op.buildTree()
	return ChunkCtxMgr{
		op: op,
	}
}

func (mgr *ChunkCtxMgr) genCols() ([]column.Column, error) {
	size := 5
	headingsCol := make([][]string, size)
	textCol := make([]string, size)
	summaryCol := make([]bool, size)
	for i := range size {
		part := &mgr.op.parts[i]
		headingsCol[i] = part.Headings
		textCol[i] = part.Heading
		summaryCol[i] = true
	}
	embedCol, err := utils.EmbedText(textCol)
	if err != nil {
		return nil, err
	}

	res := []column.Column{}
	res = append(res, utils.ColumnFromSlice("headings", headingsCol))
	res = append(res, utils.ColumnFromSlice("text", textCol))
	res = append(res, column.NewColumnFloatVector("text_dense", 1536, embedCol))
	res = append(res, utils.ColumnFromSlice("summary", summaryCol))
	return res, nil
}

func (mgr *ChunkCtxMgr) done() bool {
	for i := range mgr.op.parts {
		if !mgr.op.parts[i].Summarized {
			return false
		}
	}
	return true
}

func (mgr *ChunkCtxMgr) Next() (string, string, []model.ToolDef, bool) {
	sysprompt := `
You are a summarization assistant.Your job is to summarize the specified sections in the document

You must use the provided tool to handle this task. 
Do not output any text response. 
Do not explain anything. 
Do not provide reasoning. 
Do not return any normal message. 
Your only output must be a tool call with the correct arguments.

`
	return sysprompt, mgr.genUserPrompt(), []model.ToolDef{mgr.summaryTool()}, mgr.done()
}
func (mgr *ChunkCtxMgr) genUserPrompt() string {
	var builder strings.Builder
	parts := mgr.op.foldSection(7)
	mgr.genInstruct(parts, &builder)
	mgr.writeSection(parts, &builder)
	return builder.String()
}

func (mgr *ChunkCtxMgr) genInstruct(parts []Part, builder *strings.Builder) string {
	builder.WriteString(`
You are given a document with multiple sections, and a list of section you are required to summarize..
Please follow the instructions to summarize each sections.

Instructions:
- **Read the entire document carefully.**
- Once you've understood the context of the whole document, focus only on the sections specified.
- For each section, provide a **concise summary** that captures the core ideas and main points.
- Keep the summaries clear, relevant, and accurate. Avoid unnecessary details or filler.
- Consider *all* paragraphs and subsections inside one section.
- Do not summarize subsections individually.
- Do not include content from outside this section.

IMPORTANT: for each section, you MUST consider all paragraphs and sub sections inside the section to summary.

Now read the following document and summarize these sections:

`)
	for i := range parts {
		part := parts[i]
		if part.Summarized || !part.canSummary() {
			continue
		}
		fmt.Fprintf(builder, "** %s **\n", part.Heading)
		fmt.Fprintf(builder, "ID: %d\n", i)
		if len(part.Child) != 0 {
			fmt.Fprintf(builder, "Sub sections:\n")
			for _, c := range part.Child {
				fmt.Fprintf(builder, "- %s\n", c.Heading)
			}
		}
		builder.WriteByte('\n')
	}
	return builder.String()
}

func (mgr *ChunkCtxMgr) writeSection(parts []Part, builder *strings.Builder) {
	builder.WriteString("<DOCUMENT>\n")
	i := 0
	for i < len(parts) {
		part := &parts[i]
		if part.Summarized {
			builder.WriteString(part.Heading)
			builder.WriteString("\n")
			builder.WriteString(part.Summary)
			builder.WriteString("\n")
			i = part.Next
		} else {
			builder.WriteString(part.Heading)
			builder.WriteString("\n")
			builder.WriteString(part.Content)
			builder.WriteString("\n")
			i++
		}
	}
	builder.WriteString("</DOCUMENT>\n")
}

func (mgr *ChunkCtxMgr) summaryTool() model.ToolDef {

	var def = openai.FunctionDefinition{
		Name:   "summarize",
		Strict: true,
		Description: `
Summarize the specified section in the document
`,
		Parameters: jsonschema.Definition{
			Type:                 jsonschema.Object,
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"id": {
					Type:        jsonschema.Number,
					Description: "the id of the section",
				},
				"content": {
					Type:        jsonschema.String,
					Description: "the summary content",
				},
			},
			Required: []string{"id", "content"},
		},
	}
	handler := func(argsStr string) (string, error) {
		args := struct {
			Id      int
			Content string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return mgr.op.summary(args.Id, args.Content), nil
	}
	return model.ToolDef{FunctionDefinition: def, Handler: handler}
}
