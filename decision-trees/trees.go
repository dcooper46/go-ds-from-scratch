package main

import (
	"log"
	"math"

	"github.com/dcooper46/go-ds-from-scratch/utils"
)

// Entropy measures the uncertainty in data
// represented as a probability distribution
func Entropy(probs []float64) (entropy float64) {
	for _, p := range probs {
		if p > 0.0 {
			entropy += -p * math.Log2(p)
		}
	}
	return
}

// GetProbabilities returns label probabilities given a slice of labels (strings)
func GetProbabilities(labels []string) []float64 {
	n := float64(len(labels))
	labelCounts := make(map[string]int)
	for _, label := range labels {
		labelCounts[label]++
	}
	counts := make([]float64, len(labelCounts))
	var i int
	for _, c := range labelCounts {
		counts[i] = float64(c)
		i++
	}
	return utils.ScalarMultiply(1.0/n, counts)
}

// DataEntropy returns the entropy for a set of labels/data
func DataEntropy(data []string) float64 {
	probs := GetProbabilities(data)
	return Entropy(probs)
}

// PartitionEntropy finds the entropy of a partition of subsets of data
// where the partition entropy is a weighted sum of the entropies of
// each subset
//
// H = q1*H(S1) + q2*H(S2) + ... + qn*H(Sn) where qi is the subset proportion
func PartitionEntropy(subsets [][]string) (H float64) {
	var N float64
	for _, s := range subsets {
		N += float64(len(s))
	}
	for _, s := range subsets {
		H += DataEntropy(s) * float64(len(s)) / N
	}
	return
}

// PartitionBy groups data by values of an attribute
//
// here it is assumed the desired label is the last element in each record
func PartitionBy(data [][]string, attribute int) map[string][]string {
	if attribute >= len(data[0])-1 {
		log.Fatalf("attribute index outside of possible values, or label index")
	}
	groups := make(map[string][]string)
	for _, row := range data {
		groups[row[attribute]] = append(groups[row[attribute]], row[len(row)-1])
	}
	return groups
}

// GetPartitionEntropy returns the entropy of data partitioned by a specific attribute
func GetPartitionEntropy(data [][]string, attribute int) float64 {
	partitions := PartitionBy(data, attribute)
	subsets := make([][]string, len(partitions))

}
