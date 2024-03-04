package main

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/joho/godotenv"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"log"
	"net/http"
	"os"
)

type VC []float32

var endpoint = "http://0.0.0.0:8000"

func main() {
	err := godotenv.Load("../.env")
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	v := getJson("https://www.shihjie.com/upload/data/test/goldfish/n01443537_3883.JPEG", "")

	CollectionName := "reverse_image_search"
	vectorField := "embedding"
	MILVUS_URI := os.Getenv("MILVUS_URI")
	MILVUS_TOKEN := os.Getenv("MILVUS_TOKEN")

	milvusClient, err := client.NewClient(
		context.Background(), // ctx
		client.Config{
			Address: MILVUS_URI,
			APIKey:  MILVUS_TOKEN,
		},
	)
	if err != nil {
		log.Fatal("failed to connect to Milvus:", err.Error())
	}

	err = milvusClient.LoadCollection(
		context.Background(), // ctx
		CollectionName,       // CollectionName
		false,                // async
	)
	if err != nil {
		log.Fatal("failed to load collection:", err.Error())
	}

	// search
	sp, _ := entity.NewIndexIvfFlatSearchParam( // NewIndex*SearchParam func
		10, // searchParam
	)

	opt := client.SearchQueryOptionFunc(func(option *client.SearchQueryOption) {
		option.Limit = 3
		option.Offset = 0
		option.ConsistencyLevel = entity.ClStrong
		option.IgnoreGrowing = false
	})

	searchResult, err := milvusClient.Search(
		context.Background(),                 // ctx
		CollectionName,                       // CollectionName
		[]string{},                           // partitionNames
		"",                                   // expr
		[]string{"seq", "embedding", "path"}, // outputFields
		//[]entity.Vector{entity.FloatVector([]float32{0.1, 0.2})}, // vectors
		[]entity.Vector{entity.FloatVector(v)}, // vectors
		vectorField,                            // vectorField
		entity.L2,                              // metricType
		10,                                     // topK
		sp,                                     // sp
		opt,
	)
	if err != nil {
		log.Fatal("fail to search collection:", err.Error())
	}

	fmt.Printf("%#v\n", searchResult)
	for _, sr := range searchResult {
		fmt.Println("IDs:", sr.IDs)
		fmt.Println("Scores:", sr.Scores)
		fmt.Println("seq:", sr.Fields.GetColumn("seq"))
		//fmt.Println("embedding:", sr.Fields.GetColumn("embedding"))
		fmt.Println("path:", sr.Fields.GetColumn("path"))
	}

	milvusClient.Close()
}

func getJson(url, kind string) VC {
	var vc VC
	apiUrl := endpoint + "/get-image-vec?image=" + url
	if kind == "video" {
		apiUrl = endpoint + "/get-video-vec?video=" + url
	}

	request, error := http.NewRequest("GET", apiUrl, nil)

	if error != nil {
		fmt.Println(error)
	}

	request.Header.Set("Content-Type", "application/json; charset=utf-8")

	client := &http.Client{}
	response, error := client.Do(request)

	if error != nil {
		fmt.Println(error)
	}

	err := json.NewDecoder(response.Body).Decode(&vc)
	if err != nil {
		fmt.Println(error)
	}
	// clean up memory after execution
	defer response.Body.Close()
	return vc
}
