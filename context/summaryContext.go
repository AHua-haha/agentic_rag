package context

import (
	"bytes"
	"encoding/json"
	"fmt"
	"llm_dev/model"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/rs/zerolog/log"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

var docToc = openai.FunctionDefinition{
	Name:   "record_section_info",
	Strict: true,
	Description: `
Identify and record the section hierarchy of the document.
You must record the section level and range.

IMPORTANT: NEVER record the section infomation if you do not see the end of the section.
IMPORTANT: make sure the section level is consistency.
`,
	Parameters: jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"level": {
				Type:        jsonschema.Number,
				Description: "the level of the section, start from 1",
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

type Node struct {
	idx        int
	summary    string
	summarized bool

	isHeading bool
	level     int
	start     int
	end       int
}

type Summary struct {
	Content    string
	ChunkRange [2]int
}

type Section struct {
	Name  string
	Level int
	Start int
	End   int
}

func (sec *Section) toString() string {
	return fmt.Sprintf("%s %s (start: %d, end: %d)", strings.Repeat("  ", sec.Level), sec.Name, sec.Start, sec.End)
}

type SummaryCtxMgr struct {
	chunkIdx int
	chunks   []string
	docStack []Summary

	docNode []Node

	docToc []Section

	lastEnd int
}

func NewSummaryMgr(file string) SummaryCtxMgr {
	mgr := SummaryCtxMgr{}
	mgr.loadFile(file)
	return mgr
}

func getHeadingLevel(s string) int {
	// Trim leading whitespace
	s = strings.TrimSpace(s)

	// Regular expression to match Markdown headings (one or more '#' followed by a space)
	re := regexp.MustCompile(`^#{1,6} `)
	if re.MatchString(s) {
		// Count the number of `#` characters at the beginning
		level := strings.Count(s, "#")
		return level
	}
	return 0 // return 0 if it's not a heading
}

func (mgr *SummaryCtxMgr) genNode() {
	mgr.docNode = make([]Node, len(mgr.chunks))
	headings := []*Node{}
	for i, chunk := range mgr.chunks {
		level := getHeadingLevel(chunk)
		if level == 0 {
			mgr.docNode[i] = Node{
				idx:        i,
				summarized: false,
			}
		} else {
			mgr.docNode[i] = Node{
				idx:        i,
				summarized: false,

				isHeading: true,
				level:     level,
				start:     i,
			}
			p := len(headings) - 1
			for ; p >= 0; p-- {
				if headings[p].level >= level {
					headings[p].end = i - 1
				} else {
					break
				}
			}
			headings = headings[:p+1]
			headings = append(headings, &mgr.docNode[i])
		}
	}
	for _, node := range headings {
		node.end = len(mgr.chunks) - 1
	}
}

var sysprompt = `
You are a document-chunking agent. Your task is to split the provided document into
semantically coherent chunks.
`

var sectionTreesysPrompt = `
You are a expert document processing agent, your job is to read the document and understand the document hierarchy.
Identify each section range and hierarchy. Use tool to record the section infomation.

IMPORTANT: DO NOT output any of your reasoning or explaination, just use the tools.
IMPORTANT: NEVER record the section information if the section are not fully shown in the document excerpt.
`

func (mgr *SummaryCtxMgr) Next() (string, string, []model.ToolDef, bool) {

	return sysprompt, mgr.userPrompt(), []model.ToolDef{}, mgr.lastEnd == len(mgr.chunks)
}

func (mgr *SummaryCtxMgr) content(builder *strings.Builder, window int) {
	foldedSection := []*Section{}
	size := len(mgr.docToc)
	for i := 0; i < size; {
		ptr := &mgr.docToc[i]
		foldedSection = append(foldedSection, ptr)
		i++
		for i < size && mgr.docToc[i].Start <= ptr.End {
			i++
		}
	}
	start := 0
	for _, sec := range foldedSection {
		for i := start; i < sec.Start; i++ {
			fmt.Fprintf(builder, "[%d]: %s\n", i, mgr.chunks[i])
		}
		fmt.Fprintf(builder, "[%d-%d]: section: %s (level %d) <FOLDED>\n\n", sec.Start, sec.End, sec.Name, sec.Level)
		start = sec.End + 1
	}
	end := min(start+window, len(mgr.chunks))
	for i := start; i < end; i++ {
		fmt.Fprintf(builder, "[%d]: %s\n", i, mgr.chunks[i])
	}
	fmt.Fprintf(builder, "\n** %d paragraphs remaining **\n", len(mgr.chunks)-end)
}
func (mgr *SummaryCtxMgr) userPrompt() string {
	var builder strings.Builder
	builder.WriteString(`
Rules for chunking:
1. A chunk must represent a coherent idea, topic, or subtopic.
2. Do NOT split in the middle of a sentence or logical argument.
3. Prefer splitting at headings, subheadings, major paragraph breaks, or topic shifts.
4. Ideal chunk size: 300-600 words (or ~500-1200 tokens).
5. If a section is very long, split it into multiple coherent sub-chunks.
6. Preserve the original order of the text.

Output format:
id: 22
start char index: 345
end char index: 399

id: 23
start char index: 450
end char index: 550

Now chunk the following document:


`)
	data, _ := os.ReadFile("/root/workspace/agentic_rag/MinerU_2307.09288v2__20251127030211.md")
	builder.WriteString("<DOCUMENT>\n")
	builder.Write(data[:5000])
	builder.WriteString("</DOCUMENT>\n")
	return builder.String()
}

func (mgr *SummaryCtxMgr) addSection(toc Section) string {
	for _, item := range mgr.docToc {
		if toc.Start < item.Start && toc.End >= item.End {
			continue
		}
		if toc.Start > item.Start && toc.End <= item.End {
			continue
		}
		if toc.Start > item.End || toc.End < item.Start {
			continue
		}
		return fmt.Sprintf("section range is invalid, section %s and %s range is overlap", toc.Name, item.Name)
	}
	mgr.docToc = append(mgr.docToc, toc)
	sort.Slice(mgr.docToc, func(i, j int) bool {
		return mgr.docToc[i].Start < mgr.docToc[j].Start
	})
	return "add section success"
}

func (mgr *SummaryCtxMgr) summary(idx int, content string) string {
	if idx >= len(mgr.docNode) {
		return fmt.Sprintf("index %d out of range [0, %d]", idx, len(mgr.docNode)-1)
	}
	if mgr.docNode[idx].summarized {
		return fmt.Sprintf("index %d have been summarized already", idx)
	}
	mgr.docNode[idx].summarized = true
	mgr.docNode[idx].summary = content
	return fmt.Sprintf("summarize index %d success", idx)
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

func (mgr *SummaryCtxMgr) tocTool() model.ToolDef {
	handler := func(argsStr string) (string, error) {
		args := struct {
			Level int
			Name  string
			Start int
			End   int
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}

		return mgr.addSection(Section{
			Level: args.Level,
			Name:  args.Name,
			Start: args.Start,
			End:   args.End,
		}), nil
	}
	return model.ToolDef{FunctionDefinition: docToc, Handler: handler}
}
func (mgr *SummaryCtxMgr) summaryTool() model.ToolDef {
	handler := func(argsStr string) (string, error) {
		args := struct {
			Start   string
			End     string
			Content string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return "", nil
	}
	return model.ToolDef{FunctionDefinition: summarize, Handler: handler}
}
func (mgr *SummaryCtxMgr) loadTool() model.ToolDef {
	handler := func(argsStr string) (string, error) {
		args := struct {
			Number int
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return mgr.loadChunks(args.Number), nil
	}
	return model.ToolDef{FunctionDefinition: loadNext, Handler: handler}
}

func (mgr *SummaryCtxMgr) GetToolDef() []model.ToolDef {
	return nil
}

func (mgr *SummaryCtxMgr) WriteContext(buf *bytes.Buffer) {
}
