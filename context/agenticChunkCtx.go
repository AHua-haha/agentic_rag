package context

import (
	"bytes"
	"encoding/json"
	"fmt"
	"llm_dev/model"
	"os/exec"
	"strings"

	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

var updateTOC = openai.FunctionDefinition{
	Name:   "update_toc",
	Strict: true,
	Description: `
Update the table of content of the document.
In the process of read the document and analyze the document structure.
You MUST frequently update the table of content using this tool as soon as you get a new conclusion.

Usage:
You must record the full table of the content as far as you know.
You must record all headings with these properties:
- id: the id of the heading which represent the order and hierarchy of the heading, the id should use this format: "1", "2", "3.4", "2.3", and "Title" for the title of the document.
- content: the content of the heading
- line: the line number of the heading in the document

`,
	Parameters: jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"toc": {
				Type:        jsonschema.Array,
				Description: "the current table of content",
				Items: &jsonschema.Definition{
					Type:                 jsonschema.Object,
					AdditionalProperties: false,
					Properties: map[string]jsonschema.Definition{
						"id": {
							Type:        jsonschema.String,
							Description: "the id of the headings to represent the order and hierarchy of the section, e.g. '1', '2', '1.2', 'Title'",
						},
						"content": {
							Type:        jsonschema.String,
							Description: "the description content of the heading, do not include the numbering",
						},
						"line": {
							Type:        jsonschema.Number,
							Description: "the line number of the heading in the document",
						},
					},
					Required: []string{"id", "content", "line"},
				},
			},
		},
		Required: []string{"toc"},
	},
}

type TOCitem struct {
	Id      string
	Content string
	Line    uint
}

type AgenticChunkCtxMgr struct {
	FilePath string

	ReadBuffer string
	FileChunks file

	TOC []TOCitem
}

func (mgr *AgenticChunkCtxMgr) WriteContext(buf *bytes.Buffer) {
	buf.WriteString("[DOCUMENT CONTEXT]")

	buf.WriteString(`
You have the access to the following tools to read the document and analyze the document structure and generate the table of content for the document.
- read_content: read the content of the document based on line range to read buffer.
- grep: search text in the document using ripgrep
- record_chunk: record teh relevant chunk of content that is helpful to solve the task
- update_toc: update the table of content in the progress of analyzing the document content

IMPORTANT: consider the following useful and practical guidelines to analyze the document and generate the table of content.

Guidelines for analyzing the document:
- the document may already have a table of content at the beginning, if may begin with 'content' or other word, examine teh document table and content to understand the document structure, then analyze teh document to generate a correct table of content.
- the headings may have special format like '# heading', '## heading', '** heading **', use 'grep' to search for these special format to identify the headings.
- the document original table of content may indicate the headings content like 'abstract', 'introduction', 'references', 'method', search headings by content use 'grep'.
- record the relevant content chunks to help analyze teh document structure, like the document original table of content, and the headings you found.

IMPORTANT: you MUST frequently update the table of content as soon as you get something new.
IMPORTANT: you MUST frequently record the useful file content chunks use 'record_chunk' to avoid repeat read the same content in file.

`)

	buf.WriteString("# Recorded chunks\n\n")
	mgr.FileChunks.write(buf, mgr.FilePath)

	buf.WriteString("# Read Buffer\n\n")
	buf.WriteString(mgr.ReadBuffer)

	buf.WriteString("# Current TOC:\n")
	buf.WriteString("```\n")
	for _, item := range mgr.TOC {
		buf.WriteString(fmt.Sprintf("%s %s (line %d)\n", item.Id, item.Content, item.Line))
	}
	buf.WriteString("```\n")
	buf.WriteString("[END OF DOCUMENT CONTEXT]\n")
}

func (mgr *AgenticChunkCtxMgr) grep(arguments string) string {
	rgStr := "rg " + arguments
	cmd := exec.Command("bash", "-c", rgStr)
	output, err := cmd.Output()
	var buf bytes.Buffer
	if err != nil {
		return fmt.Sprintf("Execute grep failed, Error:%s", err)
	}
	buf.WriteString("Grep result:\n\n```\n")
	buf.Write(output)
	buf.WriteString("```\n")
	return buf.String()
}

func (mgr *AgenticChunkCtxMgr) readContent(file string, line uint, mode string) string {
	if file != mgr.FilePath {
		return fmt.Sprintf("you can not read file %s, you can only read file %s", file, mgr.FilePath)
	}
	var s, e int
	if mode == "before" {
		e = int(line)
		s = max(1, e-199)
	} else if mode == "after" {
		s = int(line)
		e = s + 199
	} else if mode == "middle" {
		s = max(int(line)-100, 1)
		e = int(line) + 100
	}
	optios := fmt.Sprintf(`NR>=%d && NR<=%d {print NR ": " $0}`, s, e)
	cmd := exec.Command("awk", optios, file)
	output, err := cmd.Output()
	if err != nil {
		return fmt.Sprintf("Read lines %d-%d in file %s failed", s, e, file)
	}
	var buf bytes.Buffer
	lineCount, _ := exec.Command("wc", "-l", file).Output()
	number := strings.Fields(string(lineCount))[0]
	buf.WriteString(fmt.Sprintf("Read file %s, file total line count %s\n", file, number))
	buf.WriteString("```\n")
	buf.Write(output)
	buf.WriteString("```\n")
	mgr.ReadBuffer = buf.String()
	var responseBuilder strings.Builder
	responseBuilder.WriteString(fmt.Sprintf("Read lines %d-%d in file %s success\n", s, e, file))
	responseBuilder.WriteString(`
IMPORTANT: 'read_content' will read teh file content to the read buffer, and the previous content will be removed, you MUST record the relevant content if it is helpful to solve the task.
`)

	return responseBuilder.String()
}
func (mgr *AgenticChunkCtxMgr) GetToolDef() []model.ToolDef {
	readHandler := func(argsStr string) (string, error) {
		args := struct {
			File string
			Line uint
			Mode string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return mgr.readContent(args.File, args.Line, args.Mode), nil
	}
	grepHandler := func(argsStr string) (string, error) {
		args := struct {
			Arguments string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		return mgr.grep(args.Arguments), nil
	}
	recordChunkHandler := func(argsStr string) (string, error) {
		args := struct {
			File      string
			Startline uint
			Endline   uint
			Comment   string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		if args.File != mgr.FilePath {
			return fmt.Sprintf("You can not record file %s, you can only record content in file %s", args.File, mgr.FilePath), nil
		}
		mgr.FileChunks.addChunk(args.Startline, args.Endline, args.Comment)
		return fmt.Sprintf("record line %d-%d in file %s", args.Startline, args.Endline, args.File), nil
	}
	updateTOCHandler := func(argsStr string) (string, error) {
		args := struct {
			Toc []TOCitem
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		mgr.TOC = args.Toc
		return "update table of content success", nil
	}
	res := []model.ToolDef{
		{FunctionDefinition: readContent, Handler: readHandler},
		{FunctionDefinition: grep, Handler: grepHandler},
		{FunctionDefinition: recordChunk, Handler: recordChunkHandler},
		{FunctionDefinition: updateTOC, Handler: updateTOCHandler},
	}
	return res
}
