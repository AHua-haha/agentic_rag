package context

import "testing"

func TestSummaryCtxMgr_genNode(t *testing.T) {
	t.Run("test gen node", func(t *testing.T) {
		mgr := NewSummaryMgr("/root/workspace/agentic_rag/MinerU_2307.09288v2__20251127030211.md")
		mgr.genNode()
	})
}
