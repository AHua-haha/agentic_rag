package context

import (
	"encoding/json"
	"fmt"
	"llm_dev/utils"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/rs/zerolog/log"
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

func (mgr *RetrievalCtxMgr) search(text string, headings []string) {
	str, _ := json.Marshal(headings)
	var filter string
	if headings == nil {
		filter = ""
	} else {
		filter = fmt.Sprintf("headings == %s", str)
	}
	err := mgr.db.Search(text, 5, filter, []string{"text"}, func(results []milvusclient.ResultSet) {
		if len(results) != 1 {
			return
		}
		res := &results[0]
		fmt.Printf("res.Fields: %v\n", res.Fields)
		textCol, ok := res.GetColumn("text").(*column.ColumnVarChar)
		if !ok {
			return
		}
		for _, text := range textCol.Data() {
			fmt.Printf("text: %v\n", text)
		}
	})
	if err != nil {
		log.Error().Err(err).Msg("")
		return
	}
}
