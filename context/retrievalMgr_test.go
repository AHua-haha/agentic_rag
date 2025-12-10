package context

import (
	"fmt"
	"llm_dev/utils"
	"testing"

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
		mgr.query([]string{
			"# LLAMA 2: Open Foundation and Fine-Tuned Chat Models",
			"## 4 Safety",
			"### 4.2 Safety Fine-Tuning",
		})
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
		mgr.search("what is the table of content of the document", nil)
	})
}
