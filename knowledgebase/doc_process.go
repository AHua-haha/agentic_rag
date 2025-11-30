package knowledgebase

import (
	"encoding/json"
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

type DocSection struct {
	Id         int
	Headings   []string
	Chunks     []DocChunk
	Subsection []string
	Summary    string
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

	re := regexp.MustCompile(`\n{2,}`)

	idCount := 1
	for i := range res {
		res[i].Headings = json_obj[i].Headings
		res[i].Id = i
		text := strings.TrimSpace(json_obj[i].Content)
		paragraphs := re.Split(text, -1)
		chunks := make([]DocChunk, len(paragraphs))
		for i, str := range paragraphs {
			chunks[i].Content = str
			chunks[i].Id = idCount
			idCount++
		}
		res[i].Chunks = chunks
	}
	return res
}
