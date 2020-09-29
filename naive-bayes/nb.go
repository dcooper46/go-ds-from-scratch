package main

import (
	"log"
	"regexp"
	"strings"
)

// EXISTS is a dummy value for sets
var EXISTS = struct{}{}

type record struct {
	message string
	hit     bool
}

type wordprob struct {
	word     string
	hitprob  float64
	missprob float64
}

// Tokenize breaks a text string into unique alphanumeric words
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

// CountWords counts the occurence of each unique word
// in a training corpus
func CountWords(records []record) map[string]map[string]int {
	counts := make(map[string]map[string]int)

	for _, rec := range records {
		for _, word := range Tokenize(rec.message) {
			wordcounts := counts[word]
			if wordcounts == nil {
				counts[word] = map[string]int{}
			}
			if rec.hit {
				counts[word]["hit"]++
			} else {
				counts[word]["miss"]++
			}
		}
	}
	return counts
}

// WordProbs calculates the probability of a `hit` and `miss`
// for each work in the corpus.
func WordProbs(counts map[string]ma[string]int, nhits, nmisses int, k float64) {
	if k == 0.0 {
		k = 0.5
	}
	wordprobs := make([]wordprob, len(counts))
	var i int
	for word, counts := range counts {
		wordprobs[i] = wordprob{
			word:     word,
			hitprob:  (float64(counts["hit"]) + k) / (float64(nhits) + 2*k),
			missprob: (float64(counts["miss"]) + k) / (float64(nmisses) + 2*k),
		}
		i++
	}
	return wordprobs
}
