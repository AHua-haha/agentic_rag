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
		op.buildTree()
		for _, p := range op.parts {
			fmt.Printf("p.Heading: %v\n", p.Heading)
			for _, c := range p.Child {
				fmt.Printf("  %s\n", c.Heading)
			}
		}
	})
}

func TestContent(t *testing.T) {
	t.Run("test write content", func(t *testing.T) {
		mgr := NewChunkCtxMgr("/root/workspace/agentic_rag/MinerU_2307.09288v2__20251127030211.md")
		str := mgr.genUserPrompt()
		fmt.Printf("%s\n", str)
	})
}
