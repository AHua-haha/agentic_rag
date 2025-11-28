package utils

import (
	"fmt"
	"testing"
)

func TestProcessFile(t *testing.T) {
	t.Run("test process file chunk", func(t *testing.T) {
		ProcessFile()
	})
}

func Test_embedText(t *testing.T) {
	t.Run("test embedding", func(t *testing.T) {
		got := embedText([]string{""})
		for _, emb := range got {
			fmt.Printf("%v\n", len(emb))
		}
	})
}

func Test_createCollection(t *testing.T) {
	InitVectorDB()
	defer CloseVectorDB()
	t.Run("test create collection", func(t *testing.T) {
		createCollection()
	})
}

func Test_insertText(t *testing.T) {
	t.Run("test insert", func(t *testing.T) {
		InitVectorDB()
		defer CloseVectorDB()
		tests := []string{
			"hello this is a test embedding text",
			"this is another",
			"ohhhh ha",
		}
		err := insertText(tests)
		if err != nil {
			fmt.Printf("err: %v\n", err)
		}
	})
}

func Test_search(t *testing.T) {
	InitVectorDB()
	defer CloseVectorDB()
	t.Run("test search", func(t *testing.T) {
		search()
	})
}

func TestBuildMgr(t *testing.T) {
	InitVectorDB()
	defer CloseVectorDB()
	t.Run("test build text vetor mgr", func(t *testing.T) {
		// TODO: construct the receiver type.
		var mgr BuildTextVectorMgr
		mgr.SemanticSearch("what is rlhf", 10)
	})
}

func TestQuerSeq(t *testing.T) {
	InitVectorDB()
	defer CloseVectorDB()
	t.Run("test query seq", func(t *testing.T) {
		// TODO: construct the receiver type.
		var mgr BuildTextVectorMgr
		got := mgr.QuerySeq([]int{2, 33, 3, 5})
		fmt.Printf("len(got): %v\n", len(got))
		for _, text := range got {
			fmt.Printf("text.Seq: %v\n", text.Seq)
		}
	})
}
