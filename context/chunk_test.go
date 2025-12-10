package context

import (
	"fmt"
	"llm_dev/utils"
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
			fmt.Printf("p.Headings: %v\n", p.Headings)
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

func TestDBMgr(t *testing.T) {
	t.Run("test db mgr", func(t *testing.T) {
		mgr, err := utils.NewDBMgr()
		if err != nil {
			fmt.Printf("err: %v\n", err)
			return
		}
		defer mgr.Close()
		err = mgr.InitDB()
		if err != nil {
			fmt.Printf("err: %v\n", err)
			return
		}
		ctxMgr := NewChunkCtxMgr("/root/workspace/agentic_rag/MinerU_2307.09288v2__20251127030211.md")
		cols, err := ctxMgr.genCols()
		if err != nil {
			fmt.Printf("err: %v\n", err)
			return
		}
		err = mgr.Insert(cols)
		if err != nil {
			fmt.Printf("err: %v\n", err)
			return
		}
	})
}
