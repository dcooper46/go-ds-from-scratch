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

type email struct {
	subject string
	isSpam  bool
}

func readFiles() []email {
	// files, err := filepath.Glob(*filesDir)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// log.Printf("found %d files to process", len(files))
	// log.Println(files)

	var data []email

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
					data = append(data, email{subject: subject, isSpam: isSpam})
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

	// split data into train and test
	var train, test []email
	for _, data := range rawData {
		if rand.Float64() < 0.75 {
			train = append(train, data)
		} else {
			test = append(test, data)
		}
	}

	fmt.Printf("train: %d, test: %d\n", len(train), len(test))
	// build classifier
	// train
	// classify test data
	// count truth/predictions
}
