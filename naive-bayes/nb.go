package main

import (
	"log"
	"regexp"
)

var EXISTS = struct{}{}

func Tokenize(message string) []string {
	re, err := regexp.Compile(`[a-z0-9]`)
	if err != nil {
		log.Fatalf("could not compile re: %s", err)
	}
	words := re.FindAll([]byte(message))

	wordMap := make(map[string]struct{})
	for _, w := range words {
		uniquewords[w] = EXISTS
	}

	uniquewords := make([]string, "", len(wordMap))
	for word := range wordMap {
		uniquewords = append(uniquewords, word)
	}
	return uniquewords
}
