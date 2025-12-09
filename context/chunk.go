package context

import (
	"os"
	"regexp"
	"strings"

	"github.com/rs/zerolog/log"
)

type Part struct {
	Heading string
	Content string
	Child   []Part
}

type FileChunkOps struct {
	File  string
	parts []Part
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
		sections = append(sections, Part{
			Heading: strings.TrimSpace(string(data[h_s:h_e])),
			Content: string(data[h_e:end]),
		})
	}
	op.parts = sections
}
