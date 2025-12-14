package context

import (
	"context"
	"encoding/json"
	"fmt"
	"llm_dev/utils"
	"testing"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/rs/zerolog/log"
)

func TestRetrieval_query(t *testing.T) {
	t.Run("test query on headings", func(t *testing.T) {
		db, err := utils.NewDBMgr()
		if err != nil {
			log.Error().Err(err)
			return
		}
		defer db.Close()
		mgr := RetrievalCtxMgr{
			db: db,
		}
		// ["# LLAMA 2: Open Foundation and Fine-Tuned Chat Models","## 4 Safety","### 4.2 Safety Fine-Tuning","#### 4.2.2 Safety Supervised Fine-Tuning"]
		mgr.query("# LLAMA 2: Open Foundation and Fine-Tuned Chat Models")
	})
}

func TestRetrieval_search(t *testing.T) {
	t.Run("test vector search", func(t *testing.T) {
		db, err := utils.NewDBMgr()
		if err != nil {
			log.Error().Err(err).Msg("")
			return
		}
		defer db.Close()
		mgr := RetrievalCtxMgr{
			db: db,
		}
		// ["# LLAMA 2: Open Foundation and Fine-Tuned Chat Models","## 4 Safety","### 4.2 Safety Fine-Tuning","#### 4.2.2 Safety Supervised Fine-Tuning"]
		fmt.Printf("hello \n")
		res := mgr.searchSummary("what does this document mainly talk about?")
		for _, r := range res {
			fmt.Printf("%v\n", r)
		}
	})
}

func TestUpsertField(t *testing.T) {
	t.Run("test update a new field", func(t *testing.T) {
		db, err := utils.NewDBMgr()
		if err != nil {
			log.Error().Err(err).Msg("")
			return
		}
		defer db.Close()
		var idCol *column.ColumnInt64
		var headingCol []string
		err = db.Query("", []string{"id", "headings"}, func(result *milvusclient.ResultSet) {
			idCol = result.GetColumn("id").(*column.ColumnInt64)
			headings := result.GetColumn("headings").(*column.ColumnDynamic)
			for _, h := range headings.Data() {
				var temp struct {
					Headings []string
				}
				json.Unmarshal(h, &temp)
				headingCol = append(headingCol, temp.Headings[len(temp.Headings)-1])
			}
		})
		if err != nil {
			log.Error().Err(err).Msg("")
			return
		}
		fmt.Printf("idCol.Data(): %v\n", idCol.Data())
		fmt.Printf("headingCol: %v\n", headingCol)
		_, err = db.Client.Upsert(context.TODO(), milvusclient.NewColumnBasedInsertOption("agentic_rag").
			WithColumns(idCol, utils.ColumnFromSlice("heading", headingCol)).
			WithPartialUpdate(true))
		if err != nil {
			log.Error().Err(err).Msg("")
			return
		}
	})
}
