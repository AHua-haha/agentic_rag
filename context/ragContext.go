package context

import (
	"bytes"
	"encoding/json"
	"fmt"
	"llm_dev/model"
	"llm_dev/utils"
	"sort"
	"strings"

	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

var recordText = openai.FunctionDefinition{
	Name:   "record_text_chunk",
	Strict: true,
	Description: `
Record the text chunks that is relevant and helpful.
`,
	Parameters: jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"seqs": {
				Type:        jsonschema.Array,
				Description: "the sequence number array of the text chunks to record",
				Items: &jsonschema.Definition{
					Type: jsonschema.Number,
				},
			},
		},
		Required: []string{"seqs"},
	},
}
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
# Tool Usage

You have access to the following tools:
- semantic_search: use vector search to match the text that is semantic relevant to the query.
- search_by_seq: search by the text chunk sequence number, used when you want to get text content arround some text chunk

# Guidelines
- analyze and understand the core concept of each text chunks.
- if one text chunk mention other concept not in the text chunk, you should try to figure out what that concept is about using tools.
- if one text chunk refer to some relevant content, you should MUST try to load the relevant text.
- ALWAYS record the text chunk that is relevant and helpful to answer the user's question.
- in the process of searching relevant content, you record all the relevant and helpful text, answer the question based on all the recorded chunks.

# Examples of How to Search

<example>
--- CHUNK START ---
Seq: 19
Section: h1, h2
Text:
When describing an object, we usually mention its attributes, such as size, weight, and position. 
You can use these attributes as fields in a collection. Each field has various constraining properties, such as the data type and the dimensionality of a vector field. 
You can form a collection schema by creating the fields and defining their order. For possible applicable data types, refer to Schema Explained.
--- CHUNK END ---
analyze the core concept: collection schema, collection fields.
this text refers: 'possible applicable data types, Schema Explained'.
You can search content about 'Schema Explained'.
</example>

<example>
--- CHUNK START ---
Seq: 24
Section: h1, h2
Text:
When evaluating performance, it is crucial to balance build time, query per second (QPS), and recall rate. The general rules are as follows:
--- CHUNK END ---
analyze the core concept: evaluating performance
this text refers: 'general rules are as follow'
You can search the next text chunk using the sequence number 25 to get 'rules are as follow'
</example>

<example>
--- CHUNK START ---
Seq: 35
Section: h1, h2
Text:
The above operation is relatively time-consuming because embedding takes time. This step takes around 2 minutes using the CPU on a 2023 MacBook Pro and will be much faster with dedicated GPUs. Take a break and enjoy a cup of coffee!
--- CHUNK END ---
this text refers: 'above operation'
You can search the previous text chunk using the sequence number 34 to get 'above operation'.
</example>

`

func writeTextChunk(c utils.TextChunk) string {
	var strBuilder strings.Builder
	strBuilder.WriteString("--- CHUNK START ---\n")
	strBuilder.WriteString(fmt.Sprintf("Seq: %d\n", c.Seq))
	strBuilder.WriteString(fmt.Sprintf("Section: %s\n", strings.Join(c.Headings, ",")))
	strBuilder.WriteString("Text\n:")
	strBuilder.WriteString(c.Content)
	strBuilder.WriteString("--- CHUNK END ---\n")
	strBuilder.WriteByte('\n')
	return strBuilder.String()
}

type KnowledgeBaseMgr struct {
	BuildTextVecMgr *utils.BuildTextVectorMgr

	files []string

	recordText   []utils.TextChunk
	searchResult strings.Builder
}

func NewKnowledgeBase(file ...string) KnowledgeBaseMgr {
	return KnowledgeBaseMgr{
		files:           file,
		BuildTextVecMgr: &utils.BuildTextVectorMgr{},
	}

}

func (kb *KnowledgeBaseMgr) writeRecorded(buf *bytes.Buffer) {
	buf.WriteString("# Recorded Text Chunks\n\n")
	sort.Slice(kb.recordText, func(i, j int) bool {
		return kb.recordText[i].Seq < kb.recordText[j].Seq
	})
	textByheading := make(map[string][]*utils.TextChunk)
	for _, text := range kb.recordText {
		key := strings.Join(text.Headings, " ")
		textByheading[key] = append(textByheading[key], &text)
	}
	preHead := []string{}
	for _, texts := range textByheading {
		headings := texts[0].Headings

		for i := range headings {
			if i < len(preHead) && headings[i] == preHead[i] {
				continue
			}
			buf.WriteString(fmt.Sprintf("%s %s\n\n", strings.Repeat("#", i+1), headings[i]))
		}
		for _, text := range texts {
			buf.WriteString(writeTextChunk(*text))
		}
		preHead = headings
	}
}

func (kb *KnowledgeBaseMgr) WriteContext(buf *bytes.Buffer) {
	buf.WriteString("[KNOWLEDGE BASE]\n\n")
	buf.WriteString("The knowledge base has the following files:\n")
	for _, file := range kb.files {
		buf.WriteString(fmt.Sprintf("- File: %s\n", file))
	}
	buf.WriteString(ragPrompt)
	kb.writeRecorded(buf)
	buf.WriteString("# Query Result\n\n")
	buf.WriteString(kb.searchResult.String())
	buf.WriteString(`
IMPORTANT: you MUST examine all the query result text chunks, and record the chunks that is relevant and helpful for user's question.
IMPORTANT: the query result that you do not record will be droped.
`)
	buf.WriteString("[END OF KNOWLEDGE BASE]\n\n")
	kb.searchResult.Reset()
}

func (kb *KnowledgeBaseMgr) GetToolDef() []model.ToolDef {
	recordHandler := func(argsStr string) (string, error) {
		args := struct {
			Seqs []int
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return kb.recordChunk(args.Seqs), nil
	}
	semanticHandler := func(argsStr string) (string, error) {
		args := struct {
			Query string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return kb.semanticSearch(args.Query, nil), nil
	}
	searchSeqHandler := func(argsStr string) (string, error) {
		args := struct {
			Seqs []int
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return kb.searchBySeq(args.Seqs), nil
	}
	res := []model.ToolDef{
		{FunctionDefinition: semanticSearch, Handler: semanticHandler},
		{FunctionDefinition: searchBySeq, Handler: searchSeqHandler},
		{FunctionDefinition: recordText, Handler: recordHandler},
	}
	return res
}

func (kb *KnowledgeBaseMgr) recordChunk(seqs []int) string {
	matchText := kb.BuildTextVecMgr.QuerySeq(seqs)
	kb.recordText = append(kb.recordText, matchText...)
	return fmt.Sprintf("record text chunks of seqs %v success", seqs)
}
func (kb *KnowledgeBaseMgr) searchBySeq(seqs []int) string {
	matchText := kb.BuildTextVecMgr.QuerySeq(seqs)
	kb.searchResult.WriteString(fmt.Sprintf("Query by Seqs: %b\n", seqs))
	kb.searchResult.WriteString("Result:\n")
	kb.searchResult.WriteString("```\n")
	for _, text := range matchText {
		kb.searchResult.WriteString(writeTextChunk(text))
	}
	kb.searchResult.WriteString("```\n")
	return "search text chunk by sequence number success"
}
func (kb *KnowledgeBaseMgr) semanticSearch(query string, headings []string) string {
	matchText := kb.BuildTextVecMgr.SemanticSearch(query, 10, headings...)
	kb.searchResult.WriteString(fmt.Sprintf("Semantic Query: %s\n", query))
	kb.searchResult.WriteString("Result:\n")
	kb.searchResult.WriteString("```\n")
	for _, text := range matchText {
		kb.searchResult.WriteString(writeTextChunk(text))
	}
	kb.searchResult.WriteString("```\n")
	return fmt.Sprintf("match %d text chunk for query", len(matchText))
}
