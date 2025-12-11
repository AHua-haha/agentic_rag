package context

import (
	"encoding/json"
	"fmt"
	"llm_dev/model"
	"llm_dev/utils"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/rs/zerolog/log"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

type Action struct {
	Argument string
	Result   []ActionRes
}
type ActionRes interface {
	Content() string
}

type DocChunk struct {
	text     string
	headings []string
	Seq      int
}

func (c *DocChunk) Content() string {
	return ""
}

type Summary struct {
	text       string
	headings   []string
	heading    string
	subSection []string
}

func (s *Summary) Content() string {
	return ""
}

type Thought struct {
	Content string
	Status  string
}
type Observation struct {
	Statement string
	Ref       []ActionRes
}

type RetrievalCtxMgr struct {
	db *utils.DBmgr
}

func (mgr *RetrievalCtxMgr) query(headings string) []ActionRes {
	filter := fmt.Sprintf(`heading == "%s"`, headings)
	var matches []ActionRes
	mgr.db.Query(filter, []string{"text"}, func(result *milvusclient.ResultSet) {
		textCol, ok := result.GetColumn("text").(*column.ColumnVarChar)
		if !ok {
			return
		}
		for _, text := range textCol.Data() {
			matches = append(matches, &Summary{
				text: text,
			})
		}
	})
	return matches
}

func (mgr *RetrievalCtxMgr) searchText(text string, headings []string) []ActionRes {
	var filter string
	if headings == nil {
		filter = ""
	} else {
		str, _ := json.Marshal(headings)
		filter = fmt.Sprintf("headings == %s", str)
	}
	var matches []ActionRes
	err := mgr.db.Search(text, 5, filter, []string{"text"}, func(results []milvusclient.ResultSet) {
		if len(results) != 1 {
			return
		}
		res := &results[0]
		textCol, ok := res.GetColumn("text").(*column.ColumnVarChar)
		if !ok {
			return
		}
		for _, text := range textCol.Data() {
			matches = append(matches, &DocChunk{
				text: text,
			})
		}
	})
	if err != nil {
		log.Error().Err(err).Msg("")
		return nil
	}
	return matches
}

func (mgr *RetrievalCtxMgr) searchTextTool() model.ToolDef {
	var def = openai.FunctionDefinition{
		Name:   "vector_search_text",
		Strict: true,
		Description: `
Summarize the specified section in the document
`,
		Parameters: jsonschema.Definition{
			Type:                 jsonschema.Object,
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"query": {
					Type:        jsonschema.String,
					Description: "the id of the section",
				},
				"headings": {
					Type:        jsonschema.Array,
					Description: "the summary content",
					Items: &jsonschema.Definition{
						Type:        jsonschema.String,
						Description: "",
					},
				},
			},
			Required: []string{"query", "headings"},
		},
	}
	handler := func(argsStr string) (string, error) {
		args := struct {
			Query    string
			Headings []string
		}{}
		err := json.Unmarshal([]byte(argsStr), &args)
		if err != nil {
			return "", err
		}
		action := Action{
			Argument: fmt.Sprintf(""),
		}
		return "", nil
	}
	return model.ToolDef{FunctionDefinition: def, Handler: handler}
}
