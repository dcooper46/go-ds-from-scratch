package main

import (
	"log"
	"regexp"
	"strings"
)

var EXISTS = struct{}{}

func Tokenize(message string) []string {
	re, err := regexp.Compile(`[a-z0-9]`)
	if err != nil {
		log.Fatalf("could not compile re: %s", err)
	}
	words := re.FindAll([]byte(strings.ToLower(message)), -1)

	wordMap := make(map[string]struct{})
	for _, w := range words {
		wordMap[w] = EXISTS
	}

	uniquewords := make([]string, "", len(wordMap))
	var i int
	for word := range wordMap {
		uniquewords[i] = word
		i++
	}
	return uniquewords
}
