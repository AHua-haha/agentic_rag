package context

import (
	"encoding/json"
	"fmt"
	"llm_dev/utils"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

type RetrievalCtxMgr struct {
	db *utils.DBmgr
}

func (mgr *RetrievalCtxMgr) query(headings []string) {
	str, _ := json.Marshal(headings)
	filter := fmt.Sprintf("headings == %s", str)
	mgr.db.Query(filter, []string{"text"}, func(result *milvusclient.ResultSet) {
		textCol, ok := result.GetColumn("text").(*column.ColumnVarChar)
		if !ok {
			fmt.Printf("not ok")
			return
		}
		for _, text := range textCol.Data() {
			fmt.Printf("text: %v\n", text)
		}
	})
}
