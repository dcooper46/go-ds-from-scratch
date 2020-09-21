package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

func checkError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func splitData(data []LabeledPoint, testPercent float64) (train, test []LabeledPoint) {
	rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
	for _, point := range data {
		if rand.Float64() < testPercent {
			test = append(test, point)
		} else {
			train = append(train, point)
		}
	}
	return train, test
}

func readIris() []LabeledPoint {
	f, err := os.Open("/Users/dancoope/go/data/iris.csv")
	checkError(err)
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 5

	rawData, err := reader.ReadAll()
	checkError(err)

	irisData := make([]LabeledPoint, len(rawData)-1)
	for i, row := range rawData {
		if i == 0 {
			continue
		}

		sepalLen, err := strconv.ParseFloat(row[0], 64)
		checkError(err)
		sepalWid, err := strconv.ParseFloat(row[1], 64)
		checkError(err)
		petalLen, err := strconv.ParseFloat(row[2], 64)
		checkError(err)
		petalWid, err := strconv.ParseFloat(row[3], 64)
		checkError(err)
		species := row[4]

		measurements := []float64{sepalLen, sepalWid, petalLen, petalWid}
		irisData[i-1] = LabeledPoint{point: measurements, label: species}
	}

	return irisData
}

func main() {
	// read iris from ~/go/data/iris.csv
	irisData := readIris()

	// split data for train/test
	train, test := splitData(irisData, 0.45)

	fmt.Printf("raw: %d, train: %d, test: %d\n", len(irisData), len(train), len(test))

	// run through test set using train set keeping track of classifications
	truth := make([]string, len(test))
	predictions := make([]string, len(test))
	for i, testRecord := range test {
		truth[i] = testRecord.label
		predictions[i] = KnnClassify(5, train, testRecord.point).label
	}

	// print confusion matrix and some metrics
	var correct int
	for i, predicted := range predictions {
		if predicted == truth[i] {
			correct++
		}
	}
	fmt.Printf("truth: %s...\n", strings.Join(truth[:5], ","))
	fmt.Printf("preds: %s...\n", strings.Join(predictions[:5], ","))
	fmt.Printf("accuracy: %f\n", float64(correct)/float64(len(predictions)))
}
