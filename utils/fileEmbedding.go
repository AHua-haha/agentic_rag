package utils

import (
	"context"
	"fmt"

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

func EmbedText(text []string) ([][]float32, error) {
	base_url := "https://openrouter.ai/api/v1"

	model := NewModel(base_url, "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")
	resp, err := model.CreateEmbeddings(context.TODO(), openai.EmbeddingRequestStrings{
		Model:          "openai/text-embedding-3-small",
		Input:          text,
		EncodingFormat: openai.EmbeddingEncodingFormatFloat,
	})
	if err != nil {
		return nil, err
	}
	res := make([][]float32, 0, len(resp.Data))
	for _, emb := range resp.Data {
		res = append(res, emb.Embedding)
	}
	if len(res) != len(text) {
		return nil, fmt.Errorf("embedding result length %d not correct", len(res))
	}
	return res, nil
}

func search() {

	var g_client *milvusclient.Client
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

func ColumnFromSlice[T any](name string, data []T) column.Column {
	switch any(data).(type) {
	case []int8:
		return column.NewColumnInt8(name, any(data).([]int8))
	case []int16:
		return column.NewColumnInt16(name, any(data).([]int16))
	case []int32:
		return column.NewColumnInt32(name, any(data).([]int32))
	case []int64:
		return column.NewColumnInt64(name, any(data).([]int64))
	case []float32:
		return column.NewColumnFloat(name, any(data).([]float32))
	case []float64:
		return column.NewColumnDouble(name, any(data).([]float64))
	case []string:
		return column.NewColumnVarChar(name, any(data).([]string))
	case [][]string:
		return column.NewColumnVarCharArray(name, any(data).([][]string))
	case [][]float32:
		return column.NewColumnFloatArray(name, any(data).([][]float32))
	case []bool:
		return column.NewColumnBool(name, any(data).([]bool))
	default:
		log.Fatal().Msgf("unsupported column type %T", data)
		return nil
	}
}

type DBmgr struct {
	client *milvusclient.Client
}

type ResHandler func(result *milvusclient.ResultSet)
type ResArrayHandler func(results []milvusclient.ResultSet)

func NewDBMgr() (*DBmgr, error) {
	milvusAddr := "172.17.0.1:19530"
	client, err := milvusclient.New(context.TODO(), &milvusclient.ClientConfig{
		Address: milvusAddr,
	})
	if err != nil {
		return nil, err
	}
	log.Info().Any("address", milvusAddr).Msg("create milvus client succes")
	return &DBmgr{client: client}, nil
}

func (db *DBmgr) Close() {
	if db.client != nil {
		db.client.Close(context.TODO())
		log.Info().Msg("close milvus client succes")
	}
}

func (db *DBmgr) InitDB() error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	exist, err := db.client.HasCollection(ctx, milvusclient.NewDescribeCollectionOption("agentic_rag"))
	if err != nil {
		return err
	}
	if exist {
		return nil
	}
	function := entity.NewFunction().
		WithName("text_bm25_emb").
		WithInputFields("text").
		WithOutputFields("text_sparse").
		WithType(entity.FunctionTypeBM25)

	schema := entity.NewSchema().WithDynamicFieldEnabled(true)

	schema.WithField(entity.NewField().
		WithName("id").
		WithDataType(entity.FieldTypeInt64).
		WithIsPrimaryKey(true).
		WithIsAutoID(true),
	).WithField(entity.NewField().
		WithName("text").
		WithDataType(entity.FieldTypeVarChar).
		WithEnableAnalyzer(true).
		WithMaxLength(3500),
	).WithField(entity.NewField().
		WithName("text_dense").
		WithDataType(entity.FieldTypeFloatVector).
		WithDim(1536),
	).WithField(entity.NewField().
		WithName("text_sparse").
		WithDataType(entity.FieldTypeSparseVector),
	).WithFunction(function).WithDynamicFieldEnabled(true)

	indexOption1 := milvusclient.NewCreateIndexOption("agentic_rag", "text_dense",
		index.NewAutoIndex(index.MetricType(entity.IP)))
	indexOption2 := milvusclient.NewCreateIndexOption("agentic_rag", "text_sparse",
		index.NewSparseInvertedIndex(entity.BM25, 0.2))

	err = db.client.CreateCollection(ctx,
		milvusclient.NewCreateCollectionOption("agentic_rag", schema).
			WithIndexOptions(indexOption1, indexOption2))

	if err != nil {
		return err
	}
	return nil
}

func (db *DBmgr) Insert(cols []column.Column) error {
	_, err := db.client.Insert(context.TODO(),
		milvusclient.NewColumnBasedInsertOption("agentic_rag").
			WithColumns(cols...),
	)
	return err
}

func (db *DBmgr) Query(filter string, fields []string, handler ResHandler) error {
	resultSet, err := db.client.Query(context.TODO(), milvusclient.NewQueryOption("agentic_rag").
		WithFilter(filter).
		WithOutputFields(fields...))
	if err != nil {
		return err
	}
	handler(&resultSet)
	return nil
}

func (db *DBmgr) Search(text string, topK int, filter string, fields []string, handler ResArrayHandler) error {
	embedding, err := EmbedText([]string{text})
	if err != nil {
		return err
	}

	resultSets, err := db.client.Search(context.TODO(), milvusclient.NewSearchOption(
		"agentic_rag",
		topK,
		[]entity.Vector{entity.FloatVector(embedding[0])},
	).WithOutputFields(fields...).
		WithANNSField("text_dense"))
	if err != nil {
		return err
	}
	handler(resultSets)
	return nil
}
