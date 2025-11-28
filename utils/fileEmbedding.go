package utils

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/rs/zerolog/log"
	"github.com/sashabaranov/go-openai"
)

type Model struct {
	*openai.Client
	apikey  string
	baseUrl string
}

func NewModel(baseurl string, apikey string) *Model {
	cfg := openai.DefaultConfig(apikey)
	cfg.BaseURL = baseurl
	return &Model{
		Client:  openai.NewClientWithConfig(cfg),
		apikey:  apikey,
		baseUrl: baseurl,
	}
}

type TextChunk struct {
	Headings []string
	Content  string
	Seq      int
}

func ProcessFile() {
	data, err := os.ReadFile("/root/workspace/agentic_rag/chunk.json")
	if err != nil {
		log.Error().Err(err).Msg("read file failed")
		return
	}

	var filechunks []TextChunk
	err = json.Unmarshal(data, &filechunks)
	if err != nil {
		log.Error().Err(err).Msg("parse json failed")
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "localhost:19530"
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
	})
	if err != nil {
		fmt.Println(err.Error())
		// handle error
	}
	defer client.Close(ctx)

}

func embedText(text []string) [][]float32 {
	base_url := "https://openrouter.ai/api/v1"

	model := NewModel(base_url, "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")
	resp, err := model.CreateEmbeddings(context.TODO(), openai.EmbeddingRequestStrings{
		Model:          "openai/text-embedding-3-small",
		Input:          text,
		EncodingFormat: openai.EmbeddingEncodingFormatFloat,
	})
	if err != nil {
		log.Error().Err(err).Msg("create embedding failed")
		return nil
	}
	var res [][]float32
	for _, emb := range resp.Data {
		res = append(res, emb.Embedding)
	}
	return res
}

var g_client *milvusclient.Client

func InitVectorDB() {
	milvusAddr := "172.17.0.1:19530"
	client, err := milvusclient.New(context.TODO(), &milvusclient.ClientConfig{
		Address: milvusAddr,
	})
	if err != nil {
		log.Error().Err(err).Msg("create client failed")
		return
	}
	g_client = client
	log.Info().Any("address", milvusAddr).Msg("create milvus client")
}

func CloseVectorDB() {
	if g_client == nil {
		return
	}
	g_client.Close(context.TODO())
	log.Info().Msg("close milvus client")
}

func createCollection() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	exist, err := g_client.HasCollection(ctx, milvusclient.NewDescribeCollectionOption("my_collection"))
	if err != nil {
		log.Error().Err(err).Msg("check collection exist failed")
		return
	}
	if exist {
		log.Error().Msg("collection exist")
		return
	}
	function := entity.NewFunction().
		WithName("text_bm25_emb").
		WithInputFields("text").
		WithOutputFields("text_sparse").
		WithType(entity.FunctionTypeBM25)

	schema := entity.NewSchema()

	schema.WithField(entity.NewField().
		WithName("id").
		WithDataType(entity.FieldTypeInt64).
		WithIsPrimaryKey(true).
		WithIsAutoID(true),
	).WithField(entity.NewField().
		WithName("text").
		WithDataType(entity.FieldTypeVarChar).
		WithEnableAnalyzer(true).
		WithMaxLength(1000),
	).WithField(entity.NewField().
		WithName("text_dense").
		WithDataType(entity.FieldTypeFloatVector).
		WithDim(1536),
	).WithField(entity.NewField().
		WithName("text_sparse").
		WithDataType(entity.FieldTypeSparseVector),
	).WithField(entity.NewField().
		WithName("seq").
		WithDataType(entity.FieldTypeInt64),
	).WithField(entity.NewField().
		WithName("headings").
		WithDataType(entity.FieldTypeArray).
		WithElementType(entity.FieldTypeVarChar).
		WithMaxCapacity(10).
		WithMaxLength(200),
	).WithFunction(function).WithDynamicFieldEnabled(true)

	indexOption1 := milvusclient.NewCreateIndexOption("my_collection", "text_dense",
		index.NewAutoIndex(index.MetricType(entity.IP)))
	indexOption2 := milvusclient.NewCreateIndexOption("my_collection", "text_sparse",
		index.NewSparseInvertedIndex(entity.BM25, 0.2))

	err = g_client.CreateCollection(ctx,
		milvusclient.NewCreateCollectionOption("my_collection", schema).
			WithIndexOptions(indexOption1, indexOption2))

	if err != nil {
		log.Error().Err(err).Msg("create collection failed")
		return
	}

}

func insertText(text []string) error {
	embeddings := embedText(text)
	if len(embeddings) != len(text) {
		return fmt.Errorf("embedding len not match text len")
	}
	_, err := g_client.Insert(context.TODO(), milvusclient.NewColumnBasedInsertOption("my_collection").
		WithVarcharColumn("text", text).
		WithFloatVectorColumn("text_dense", 1536, embeddings),
	)
	return err
}

func search() {

	annSearchParams := index.NewCustomAnnParam()
	annSearchParams.WithExtraParam("drop_ratio_search", 0.2)
	resultSets, err := g_client.Search(context.TODO(), milvusclient.NewSearchOption(
		"my_collection", // collectionName
		3,               // limit
		[]entity.Vector{entity.Text("i want to say hello")},
	).WithConsistencyLevel(entity.ClStrong).
		WithANNSField("text_sparse").
		WithOutputFields("text", "text_dense"))
	if err != nil {
		fmt.Println(err.Error())
	}
	for _, resultSet := range resultSets {
		for _, col := range resultSet.Fields {
			fmt.Printf("col.Name(): %v\n", col.Name())
			switch col := col.(type) {
			case *column.ColumnFloatVector:
				dense := col.Data()[0]
				fmt.Printf("dense: %v\n", dense)
			case *column.ColumnVarChar:
				fmt.Printf("col.Len(): %v\n", col.Len())
			}
		}
	}
}

type BuildTextVectorMgr struct {
	texts []TextChunk
}

func (mgr *BuildTextVectorMgr) readText(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Error().Err(err).Msg("read file failed")
		return
	}

	var filechunks []TextChunk
	err = json.Unmarshal(data, &filechunks)
	if err != nil {
		log.Error().Err(err).Msg("parse json failed")
		return
	}
	mgr.texts = filechunks
}

func (mgr *BuildTextVectorMgr) insert() {
	idx := 0
	step := 30
	size := len(mgr.texts)
	for {
		end := min(idx+step, size)
		part := mgr.texts[idx:end]
		var seqCol []int64
		var textCol []string
		var embedCol [][]float32
		var headingsCol [][]string
		for i, text := range part {
			textCol = append(textCol, text.Content)
			headingsCol = append(headingsCol, text.Headings)
			seqCol = append(seqCol, int64(idx+i))
		}
		col := column.NewColumnVarCharArray("headings", headingsCol)
		embedCol = embedText(textCol)
		_, err := g_client.Insert(context.TODO(), milvusclient.NewColumnBasedInsertOption("my_collection").
			WithVarcharColumn("text", textCol).
			WithFloatVectorColumn("text_dense", 1536, embedCol).
			WithInt64Column("seq", seqCol).
			WithColumns(col),
		)
		if err != nil {
			log.Error().Err(err).Msg("insert text failed")
		} else {
			log.Info().Any("size", len(textCol)).Msg("insert text success")
		}
		idx = end
		if idx >= size {
			break
		}
	}
}

func (mgr *BuildTextVectorMgr) SemanticSearch(query string, topk int, heading ...string) []TextChunk {
	embed := embedText([]string{query})
	resultSets, err := g_client.Search(context.TODO(), milvusclient.NewSearchOption(
		"my_collection", // collectionName
		topk,
		[]entity.Vector{entity.FloatVector(embed[0])},
	).WithConsistencyLevel(entity.ClStrong).
		WithANNSField("text_dense").
		WithOutputFields("text", "seq", "headings"))
	if err != nil {
		log.Error().Err(err).Msg("execute search request failed")
		return nil
	}
	if len(resultSets) != 1 {
		return nil
	}
	resultSet := resultSets[0]
	seqCol := resultSet.GetColumn("seq").(*column.ColumnInt64)
	textCol := resultSet.GetColumn("text").(*column.ColumnVarChar)
	headingsCol := resultSet.GetColumn("headings").(*column.ColumnVarCharArray)
	size := len(seqCol.Data())
	res := make([]TextChunk, 0, size)
	for i := range size {
		res = append(res, TextChunk{
			Seq:      int(seqCol.Data()[i]),
			Headings: headingsCol.Data()[i],
			Content:  textCol.Data()[i],
		})
	}
	return res
}

func (mgr *BuildTextVectorMgr) lexicalSearch(query string, topk int) []TextChunk {
	resultSets, err := g_client.Search(context.TODO(), milvusclient.NewSearchOption(
		"my_collection", // collectionName
		topk,
		[]entity.Vector{entity.Text(query)},
	).WithConsistencyLevel(entity.ClStrong).
		WithANNSField("text_sparse").
		WithOutputFields("text", "seq", "headings"))
	if err != nil {
		log.Error().Err(err).Msg("execute search request failed")
		return nil
	}
	if len(resultSets) != 1 {
		return nil
	}
	resultSet := resultSets[0]
	seqCol := resultSet.GetColumn("seq").(*column.ColumnInt64)
	textCol := resultSet.GetColumn("text").(*column.ColumnVarChar)
	headingsCol := resultSet.GetColumn("headings").(*column.ColumnVarCharArray)
	size := len(seqCol.Data())
	res := make([]TextChunk, 0, size)
	for i := range size {
		res = append(res, TextChunk{
			Seq:      int(seqCol.Data()[i]),
			Headings: headingsCol.Data()[i],
			Content:  textCol.Data()[i],
		})
	}
	return res
}

func (mgr *BuildTextVectorMgr) QuerySeq(seqs []int) []TextChunk {
	seqsStr, _ := json.Marshal(seqs)
	fileter := fmt.Sprintf("seq in %s", seqsStr)
	resultSet, err := g_client.Query(context.TODO(), milvusclient.NewQueryOption("my_collection").
		WithFilter(fileter).
		WithOutputFields("text", "seq", "headings"))
	if err != nil {
		log.Error().Err(err).Msg("execute query request failed")
		return nil
	}
	fmt.Printf("resultSet.Len(): %v\n", resultSet.Len())
	seqCol := resultSet.GetColumn("seq").(*column.ColumnInt64)
	textCol := resultSet.GetColumn("text").(*column.ColumnVarChar)
	headingsCol := resultSet.GetColumn("headings").(*column.ColumnVarCharArray)
	size := len(seqCol.Data())
	res := make([]TextChunk, 0, size)
	for i := range size {
		res = append(res, TextChunk{
			Seq:      int(seqCol.Data()[i]),
			Headings: headingsCol.Data()[i],
			Content:  textCol.Data()[i],
		})
	}
	return res
}
