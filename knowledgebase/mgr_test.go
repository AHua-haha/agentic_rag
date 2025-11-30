package knowledgebase

import (
	"fmt"
	"testing"
)

func TestReplaceAgent(t *testing.T) {
	t.Run("test doc mgr", func(t *testing.T) {
		mgr := DocMgr{}
		res := mgr.loadFile("/root/workspace/agentic_rag/doc.json")
		for i := range 4 {
			c := res[i].Chunks
			for _, chunk := range c {
				fmt.Printf("chunk.Id: %v\n", chunk.Id)
				fmt.Printf("chunk.Content: %v\n\n", chunk.Content)
			}
		}
	})
}
