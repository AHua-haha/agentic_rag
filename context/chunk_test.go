package context

import (
	"fmt"
	"testing"
)

func TestFileChunkOps_chunk(t *testing.T) {
	t.Run("test chunk file", func(t *testing.T) {
		op := FileChunkOps{
			File: "/root/workspace/agentic_rag/MinerU_2307.09288v2__20251127030211.md",
		}
		op.chunk()
		for _, p := range op.parts {
			fmt.Printf("p.Heading: %v\n", p.Heading)
		}
	})
}
