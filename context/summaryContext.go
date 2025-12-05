package context

import (
	"bytes"
	"encoding/json"
	"fmt"
	"llm_dev/model"
	"os"
	"regexp"
	"strings"

	"github.com/rs/zerolog/log"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

var docToc = openai.FunctionDefinition{
	Name:   "doc_toc",
	Strict: true,
	Description: `
Mark the section range and section name.
`,
	Parameters: jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"toc": {
				Type:        jsonschema.Array,
				Description: "the table of content of the document, each item represent one section",
				Items: &jsonschema.Definition{
					Type:                 jsonschema.Object,
					AdditionalProperties: false,
					Properties: map[string]jsonschema.Definition{
						"level": {
							Type:        jsonschema.Number,
							Description: "the level of the section, start from 0",
						},
						"name": {
							Type:        jsonschema.String,
							Description: "the name of the section",
						},
						"start": {
							Type:        jsonschema.Number,
							Description: "the start paragraph id of the section",
						},
						"end": {
							Type:        jsonschema.Number,
							Description: "the end paragraph id of the section",
						},
					},
					Required: []string{"level", "name", "start", "end"},
				},
			},
		},
		Required: []string{"toc"},
	},
}
var markSection = openai.FunctionDefinition{
	Name:   "mark_section",
	Strict: true,
	Description: `
Mark the section range and section name.
`,
	Parameters: jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"start": {
				Type:        jsonschema.Number,
				Description: "the start paragraph id of this section",
			},
			"end": {
				Type:        jsonschema.Number,
				Description: "the end paragraph id of this section",
			},
			"name": {
				Type:        jsonschema.String,
				Description: "the name of the secton",
			},
		},
		Required: []string{"start", "end", "name"},
	},
}
var loadNext = openai.FunctionDefinition{
	Name:   "load_paragraph",
	Strict: true,
	Description: `
Load the remaining paragraphs of the document.
`,
	Parameters: jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"number": {
				Type:        jsonschema.Number,
				Description: "the number of paragraphs to load",
			},
		},
		Required: []string{"number"},
	},
}
var summarize = openai.FunctionDefinition{
	Name:   "summarize",
	Strict: true,
	Description: `
Summary certain paragraphs in the document.

You can summary paragraphs and sections.

IMPORTANT: you MUST first summarize each paragraphs under one sections.
IMPORTANT: add the section headings when summarize a section.

Summary Principles:
- identify the core concepts of the paragraphs to sumarize.
- briefly describe the core concept.
- summarize how the paragraphs introduce and discuss about the core concept.
- keep summary short and concise.
- summarize a paragraph in 1-3 sentences
- summarize a whole section in 4-6 sentences
- output plain text for summary content.
`,
	Parameters: jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"start": {
				Type:        jsonschema.String,
				Description: "the start paragraphs first 8 words to match the paragraph",
			},
			"end": {
				Type:        jsonschema.String,
				Description: "the end paragraphs first 8 words to match the paragraph",
			},
			"content": {
				Type:        jsonschema.String,
				Description: "the summary content of the paragraphs",
			},
		},
		Required: []string{"start", "end", "content"},
	},
}

var sectionTreePrompt = `
*** Instruction ***

Your Job:
- read the document and understand the document structure.
- identify section boundary and section hierarchy.
- mark the section with it's name range and sub section name.


IMPORTANT: NEVER try to read the whole document, Mark the section that is already complete first.
IMPORTANT: NEVER output any of your reasoning or explaination, just use the tools to summarize the whole document.
`

var instructPrompt = `
*** Instruction ***
First summarize paragraphs. When all paragraphs under one section is summarized, summarize the section.

IMPORTANT: you MUST first summarize section paragraph, before you summarize the whole section.
IMPORTANT: DO NOT output any of your reasoning or explaination, just use the tools to summarize the whole document.

`

type Summary struct {
	Content    string
	ChunkRange [2]int
}

type Section struct {
	Name       string
	SubSection []string
	ChunkRange [2]int
}

type SummaryCtxMgr struct {
	chunkIdx int
	chunks   []string
	docStack []Summary

	sectionTree map[string]Section
}

func NewSummaryMgr(file string) SummaryCtxMgr {
	mgr := SummaryCtxMgr{
		sectionTree: map[string]Section{},
	}
	mgr.loadFile(file)
	mgr.loadChunks(5)
	return mgr
}

var sysprompt = `
You are a helpful document processing agent, you job is to build index for the content of the document for quick look up when searching in the document.
The index is the summary of the content in the document, and the values is the actual paragraphs in the document.
`

var sectionTreesysPrompt = `
You are a document processing agent, your job is to read the document and understand the document structure.
And use the tool to mark each section.
`

func (ctx *SummaryCtxMgr) GetSysPrompt() string {
	return sectionTreesysPrompt
}

func (mgr *SummaryCtxMgr) WriteContext(buf *bytes.Buffer) {
	buf.WriteString(sectionTreePrompt)
	buf.WriteString("<DOCUMENT>\n")
	for _, chunk := range mgr.docStack {
		id := chunk.ChunkRange
		if id[0] == id[1] {
			buf.WriteString(fmt.Sprintf("[%d]:", id[0]))
		} else {
			buf.WriteString(fmt.Sprintf("[%d-%d]:", id[0], id[1]))
		}
		buf.WriteString(chunk.Content)
		buf.WriteString("\n\n")
	}
	buf.WriteString(fmt.Sprintf("=== %d paragraphs remaining ===\n", len(mgr.chunks)-mgr.chunkIdx))
	buf.WriteString("</DOCUMENT>\n")
}

func (mgr *SummaryCtxMgr) markSection(name string, s, e int) string {
	if _, exist := mgr.sectionTree[name]; exist {
		return fmt.Sprintf("section '%s' already exist", name)
	}
	mgr.sectionTree[name] = Section{
		Name:       name,
		ChunkRange: [2]int{s, e},
	}
	length := len(mgr.docStack)
	stack_s, stack_e := -1, -1
	for i := length - 1; i >= 0; i-- {
		if mgr.docStack[i].ChunkRange[0] == s {
			stack_s = i
		}
		if mgr.docStack[i].ChunkRange[1] == e {
			stack_e = i
		}
	}
	if stack_s == -1 || stack_e == -1 || stack_s > stack_e {
		return fmt.Sprint("do not match paragraphs range %d-%d", s, e)
	}
	groupSummary := Summary{Content: fmt.Sprintf("Section: %s\n", name), ChunkRange: [2]int{stack_s, stack_e}}
	newStack := mgr.docStack[:stack_s]
	newStack = append(newStack, groupSummary)
	newStack = append(newStack, mgr.docStack[stack_e+1:]...)
	mgr.docStack = newStack
	return fmt.Sprintf("create section '%s' success", name)
}

func (mgr *SummaryCtxMgr) summary(s, e string, content string) string {
	length := len(mgr.docStack)
	stack_s, stack_e := -1, -1
	p_s, p_e := 0, 0
	for i := length - 1; i >= 0; i-- {
		if strings.HasPrefix(mgr.docStack[i].Content, s) {
			stack_s = i
			p_s = mgr.docStack[i].ChunkRange[0]
		}
		if strings.HasPrefix(mgr.docStack[i].Content, e) {
			stack_e = i
			p_e = mgr.docStack[i].ChunkRange[1]
		}
	}
	if stack_s == -1 || stack_e == -1 || stack_s > stack_e {
		return fmt.Sprint("do not match paragraphs range %d-%d", s, e)
	}
	groupSummary := Summary{Content: content, ChunkRange: [2]int{p_s, p_e}}
	newStack := mgr.docStack[:stack_s]
	newStack = append(newStack, groupSummary)
	newStack = append(newStack, mgr.docStack[stack_e+1:]...)
	mgr.docStack = newStack
	return fmt.Sprintf("summarize paragraph range %d-%d", s, e)
}

func (mgr *SummaryCtxMgr) loadChunks(num int) string {
	end := min(len(mgr.chunks), mgr.chunkIdx+num, mgr.chunkIdx+5)
	chunks := mgr.chunks[mgr.chunkIdx:end]

	for i, c := range chunks {
		mgr.docStack = append(mgr.docStack, Summary{
			Content:    c,
			ChunkRange: [2]int{mgr.chunkIdx + i, mgr.chunkIdx + i},
		})
	}
	mgr.chunkIdx = end
	return fmt.Sprintf("load %d paragraphs success", len(chunks))
}

func (mgr *SummaryCtxMgr) loadFile(file string) {
	data, err := os.ReadFile(file)
	if err != nil {
		log.Error().Err(err).Msg("read file failed")
		return
	}
	re := regexp.MustCompile(`\n{2,}`)
	mgr.chunks = re.Split(string(data), -1)
}

func (mgr *SummaryCtxMgr) sectionTreeTool() []model.ToolDef {

	markHandler := func(argsStr string) (string, error) {
		args := struct {
			Start int
			End   int
			Name  string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return mgr.markSection(args.Name, args.Start, args.End), nil
	}
	loadHandler := func(argsStr string) (string, error) {
		args := struct {
			Number int
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return mgr.loadChunks(args.Number), nil
	}
	res := []model.ToolDef{
		{FunctionDefinition: markSection, Handler: markHandler},
		{FunctionDefinition: loadNext, Handler: loadHandler},
	}
	return res
}

func (mgr *SummaryCtxMgr) summarizeTool() []model.ToolDef {

	summarizeHandler := func(argsStr string) (string, error) {
		args := struct {
			Start   string
			End     string
			Content string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return mgr.summary(args.Start, args.End, args.Content), nil
	}
	loadHandler := func(argsStr string) (string, error) {
		args := struct {
			Number int
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return mgr.loadChunks(args.Number), nil
	}
	res := []model.ToolDef{
		{FunctionDefinition: summarize, Handler: summarizeHandler},
		{FunctionDefinition: loadNext, Handler: loadHandler},
	}
	return res
}

func (mgr *SummaryCtxMgr) GetToolDef() []model.ToolDef {
	return mgr.sectionTreeTool()
}
