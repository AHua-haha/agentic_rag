package knowledgebase

import (
	"encoding/json"
	"fmt"
	"llm_dev/agent"
	"llm_dev/model"
	"llm_dev/utils"
	"os"
	"regexp"
	"strings"

	"github.com/rs/zerolog/log"
)

type DocChunk struct {
	Id      int
	Content string
	Summary string
}

type SplitMethod func(string) []DocChunk

type DocSection struct {
	Id          int
	Headings    []string
	PageContent string
	Chunks      []DocChunk
	Subsection  []string
	Summary     string
}

func (doc *DocSection) addSubSection(heading string) {
	for _, h := range doc.Subsection {
		if h == heading {
			return
		}
	}
	doc.Subsection = append(doc.Subsection, heading)
}

func (doc *DocSection) split(method SplitMethod) {
	doc.Chunks = method(doc.PageContent)
}

func splitMethod() SplitMethod {
	re := regexp.MustCompile(`\n{2,}`)
	idCount := 1
	method := func(pageContent string) []DocChunk {
		text := strings.TrimSpace(pageContent)
		paragraphs := re.Split(text, -1)
		chunks := make([]DocChunk, len(paragraphs))
		for i, str := range paragraphs {
			chunks[i].Content = str
			chunks[i].Id = idCount
			idCount++
		}
		return chunks
	}
	return method
}

func (mgr *DocMgr) loadFile(file string) []DocSection {
	json_obj := []struct {
		Headings []string
		Content  string
	}{}
	data, err := os.ReadFile(file)
	if err != nil {
		log.Error().Err(err).Msg("read file failed")
	}

	err = json.Unmarshal(data, &json_obj)
	if err != nil {
		log.Error().Err(err).Msg("parse json failed")
	}
	res := make([]DocSection, len(json_obj))

	for i := range res {
		res[i].Headings = json_obj[i].Headings
		res[i].Id = i
		res[i].PageContent = json_obj[i].Content
	}
	return res
}
func (mgr *DocMgr) genIndex(headings []string) string {
	length := len(headings)
	if length == 0 {
		log.Fatal().Msg("headings length is zero")
	}
	header := headings[length-1]
	return strings.Repeat("#", length) + " " + header
}

func (mgr *DocMgr) process(sections []DocSection) {
	sectionMap := make(map[string]*DocSection)
	for i := range sections {
		idx := mgr.genIndex(sections[i].Headings)
		if _, exist := sectionMap[idx]; exist {
			log.Error().Any("key", idx).Msg("key already exist")
		}
		sectionMap[idx] = &sections[i]
	}
	for i := range sections {
		doc := &sections[i]
		length := len(doc.Headings)
		for j := 1; j < length; j++ {
			headings := doc.Headings[0:j]
			idx := mgr.genIndex(headings)
			sec, exist := sectionMap[idx]
			if !exist {
				sectionMap[idx] = &DocSection{
					Headings: headings,
				}
				sec = sectionMap[idx]
			}
			sec.addSubSection(doc.Headings[j])
		}
	}
	mgr.sectionMsp = sectionMap
}

type SummaryPromptFunc func(sec *DocSection) string

func (mgr *DocMgr) summaryMethod() (SummaryPromptFunc, []model.ToolDef) {
	prompt := func(sec *DocSection) string {
		var builder strings.Builder
		builder.WriteString(`
You are an expert document summarizer. Your task is to summarize and refine content in document.

Your job:
- Identify the main concepts.
- Briefly describe the main concepts, what the main concept is.
- Summarize how the content introduce and discuss about the main concept.
- Summarize how each sub section introduce and discuss about the main concept.
- keep the summary simple, short and concise.

IMPORTANT: you MUST summarizze the document solely on the document contents.
IMPORTANT: DO NOT use markdown headings, output two paragraph, first identify and describe the main concept, second summarize how the main concept is discessed.
IMPORTANT: refer each section in this format: section ** <name> **.

`)
		length := len(sec.Headings)
		builder.WriteString("Section hierarchy:\n")
		builder.WriteString("```\n")
		for i := 1; i <= length; i++ {
			idx := mgr.genIndex(sec.Headings[0:i])
			builder.WriteString(fmt.Sprintf("%s\n", idx))
		}
		builder.WriteString("```\n")

		builder.WriteString(fmt.Sprintf("Summary this section ** %s **\n", sec.Headings[length-1]))
		builder.WriteString("<DOCUMENT>\n")
		builder.WriteString(fmt.Sprintf("%s %s\n", strings.Repeat("#", length), sec.Headings[length-1]))
		builder.WriteString(sec.PageContent)
		for _, h := range sec.Subsection {
			builder.WriteString(fmt.Sprintf("%s %s\n", strings.Repeat("#", length+1), h))
		}
		builder.WriteString("</DOCUMENT>\n")
		return builder.String()
	}
	return prompt, nil
}

func (mgr *DocMgr) genSummary(section *DocSection) {

	model := utils.NewModel("https://openrouter.ai/api/v1", "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")
	llmAgent := agent.SimpleAgent{
		BaseAgent: agent.NewBaseAgent("", *model),
	}
	llmAgent.NewUserTask("", "")
}
