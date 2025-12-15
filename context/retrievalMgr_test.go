package context

import (
	"context"
	"encoding/json"
	"fmt"
	"llm_dev/utils"
	"os"
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
		res := mgr.searchText("what does this document mainly talk about?", "## Contents")
		for _, r := range res {
			fmt.Printf("%s\n", r.Metadata())
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

func TestInsertDocChunk(t *testing.T) {
	t.Run("test insert doc chunk", func(t *testing.T) {
		data, _ := os.ReadFile("/root/workspace/agentic_rag/data.json")
		var doc []struct {
			Headings []string
			Chunks   []string
		}
		err := json.Unmarshal(data, &doc)
		if err != nil {
			log.Error().Err(err).Msg("")
			return
		}
		var textCol []string
		var headingsCol [][]string
		var seqCol []int32
		count := 1
		for _, c := range doc {
			for _, text := range c.Chunks {
				textCol = append(textCol, text)
				headingsCol = append(headingsCol, c.Headings)
				seqCol = append(seqCol, int32(count))
				count++
			}
		}
		fmt.Printf("len(textCol): %v\n", len(textCol))
		fmt.Printf("len(textCol): %v\n", len(headingsCol))
		fmt.Printf("len(textCol): %v\n", len(seqCol))
		embedCol, err := utils.EmbedText(textCol)
		if err != nil {
			log.Error().Err(err).Msg("embedding text failed")
			return
		}

		cols := []column.Column{
			utils.ColumnFromSlice("text", textCol),
			utils.ColumnFromSlice("headings", headingsCol),
			utils.ColumnFromSlice("sequence", seqCol),
			column.NewColumnFloatVector("text_dense", 1536, embedCol),
		}

		db, err := utils.NewDBMgr()
		if err != nil {
			log.Error().Err(err).Msg("create db mgr failed")
			return
		}
		defer db.Close()
		err = db.Insert(cols)
		if err != nil {
			log.Error().Err(err).Msg("insert into db error")
			return
		}
	})
}

func TestDelete(t *testing.T) {
	t.Run("test delete rows", func(t *testing.T) {
		db, err := utils.NewDBMgr()
		if err != nil {
			log.Error().Err(err).Msg("create db mgr failed")
			return
		}
		defer db.Close()
		_, err = db.Client.Delete(context.TODO(), milvusclient.NewDeleteOption("agentic_rag").WithExpr("summary IS NULL"))
		if err != nil {
			log.Error().Err(err).Msg("create db mgr failed")
			return
		}
	})
}
