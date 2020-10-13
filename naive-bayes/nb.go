package main

import (
	"log"
	"math"
	"regexp"
	"strings"
)

// EXISTS is a dummy value for sets
var EXISTS = struct{}{}

// Record contains the message we're interested in
type Record struct {
	message string
	hit     bool
}

// Wordprob holds the hit and miss probabilities for a word
type Wordprob struct {
	word     string
	hitprob  float64
	missprob float64
}

func stringInList(s string, list []string) bool {
	for _, sl := range list {
		if s == sl {
			return true
		}
	}
	return false
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
		wordMap[string(w)] = EXISTS
	}

	uniquewords := make([]string, len(wordMap))
	var i int
	for word := range wordMap {
		uniquewords[i] = word
		i++
	}
	return uniquewords
}

// CountWords counts the occurence of each unique word
// in a training corpus
func CountWords(records []Record) map[string]map[string]int {
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
func WordProbs(counts map[string]map[string]int, nhits, nmisses int, k float64) []Wordprob {
	if k == 0.0 {
		k = 0.5
	}
	wordprobs := make([]Wordprob, len(counts))
	var i int
	for word, counts := range counts {
		wordprobs[i] = Wordprob{
			word:     word,
			hitprob:  (float64(counts["hit"]) + k) / (float64(nhits) + 2*k),
			missprob: (float64(counts["miss"]) + k) / (float64(nmisses) + 2*k),
		}
		i++
	}
	return wordprobs
}

// HitProb assigns a probability to a message using
// its message and precomputed word probabilities
func HitProb(wordprobs []Wordprob, message string) float64 {
	words := Tokenize(message)
	hitLogProb, missLogProb := 0.0, 0.0

	for _, wp := range wordprobs {
		if stringInList(wp.word, words) {
			hitLogProb += math.Log(wp.hitprob)
			missLogProb += math.Log(wp.missprob)
		} else {
			hitLogProb += math.Log(1.0 - wp.hitprob)
			missLogProb += math.Log(1.0 - wp.missprob)
		}
	}

	hitProb := math.Exp(hitLogProb)
	missProb := math.Exp(missLogProb)

	return hitProb / (hitProb + missProb)
}

// NaiveBayesClassifier implements a simple naive bayes algorithm
// useful for a small binary classification problem
type NaiveBayesClassifier struct {
	k         float64
	wordprobs []Wordprob
}

// Train calculates word probabilities given a
// training set of messages
func (nb *NaiveBayesClassifier) Train(data []Record) {
	var nhits, nmisses int
	for _, rec := range data {
		if rec.hit {
			nhits++
		}
	}
	nmisses = len(data) - nhits

	wordcounts := CountWords(data)
	nb.wordprobs = WordProbs(wordcounts, nhits, nmisses, nb.k)
}

// Classify predicts whether a message should be considered
// a `hit` or a `miss`
func (nb *NaiveBayesClassifier) Classify(message string) float64 {
	return HitProb(nb.wordprobs, message)
}
