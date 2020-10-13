package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

var (
	filesDir = flag.String("files", "", "directory of spam files")
)

type prediction struct {
	subject       string
	isSpam        bool
	predictedProb float64
}

func readFiles() []Record {
	// files, err := filepath.Glob(*filesDir)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// log.Printf("found %d files to process", len(files))
	// log.Println(files)

	var data []Record

	filepath.Walk(*filesDir, func(path string, info os.FileInfo, e error) error {
		if e != nil {
			return e
		}

		if info.Mode().IsRegular() {
			if strings.HasPrefix(path, ".") {
				return nil
			}

			isSpam := !strings.Contains(path, "ham")

			openFile, err := os.Open(path)
			if err != nil {
				log.Fatal(err)
			}
			defer openFile.Close()

			scanner := bufio.NewScanner(openFile)

			for scanner.Scan() {
				text := scanner.Text()

				if strings.HasPrefix(text, "Subject: ") {
					subject := strings.Replace(text, "Subject: ", "", 1)
					data = append(data, Record{message: subject, hit: isSpam})
				}
			}
		}
		return nil
	})

	return data
}

func main() {
	flag.Parse()

	rawData := readFiles()

	fmt.Printf("found %d files\n", len(rawData))
	fmt.Println(rawData[0])

	var train, test []Record
	for _, data := range rawData {
		if rand.Float64() < 0.75 {
			train = append(train, data)
		} else {
			test = append(test, data)
		}
	}

	fmt.Printf("train: %d, test: %d\n", len(train), len(test))

	classifier := NaiveBayesClassifier{k: 0.5}

	classifier.Train(train)

	predictions := make([]prediction, len(test))
	for i, data := range test {
		predictedProb := classifier.Classify(data.message)
		predictions[i] = prediction{
			subject:       data.message,
			isSpam:        data.hit,
			predictedProb: predictedProb,
		}
	}

	counts := make(map[string]int, 4)
	for _, pred := range predictions {
		if pred.isSpam && pred.predictedProb > 0.5 {
			counts["TT"]++
		} else if pred.isSpam && pred.predictedProb <= 0.5 {
			counts["TF"]++
		} else if !pred.isSpam && pred.predictedProb > 0.5 {
			counts["FT"]++
		} else {
			counts["FF"]++
		}
	}

	fmt.Println(counts)
}
