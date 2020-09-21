package main

// Basic implementation of the K-Nearest-Neighbors algorithm
// Not generalized for different data types, but basic float

import (
	"math"
	"sort"
)

// LabeledPoint is a structure representing a row from a data source for KNN
// point is an array of float feature values
// label is a string label for the record
type LabeledPoint struct {
	point []float64
	label string
}

// basic euclidean distance
func distance(x, y []float64) float64 {
	var sum float64
	for i, xi := range x {
		sum += math.Pow(xi-y[i], 2)
	}
	return math.Sqrt(sum)
}

func countLabels(labels []string) map[string]int {
	counts := make(map[string]int)
	for _, label := range labels {
		counts[label]++
	}
	return counts
}

func mostCommon(counts map[string]int) []string {
	max := 0
	labels := make([]string, 1)
	for label, count := range counts {
		if count > max {
			max = count
			labels = []string{label}
		} else if count == max {
			labels = append(labels, label)
		} else {

		}
	}
	return labels
}

// MajorityVote returns the label of the 'majority vote' from the neighbors
// Assume labels are ordered nearest to farthest
// Ties are broken by removing the furthest neighbor and trying again
func MajorityVote(labels []string) string {
	// count the labels
	winners := mostCommon(countLabels(labels))
	// handle ties
	if len(winners) > 1 {
		return MajorityVote(labels[:len(labels)-1])
	}
	// return majority vote
	return winners[0]
}

// KnnClassify returns the label for a given point from its nearest neighbors
func KnnClassify(k int, points []LabeledPoint, point []float64) LabeledPoint {
	pointsByDistance := make([]LabeledPoint, len(points))
	copy(pointsByDistance, points)
	sort.Slice(pointsByDistance, func(i, j int) bool {
		return distance(pointsByDistance[i].point, point) < distance(pointsByDistance[j].point, point)
	})

	sortedLabels := make([]string, len(pointsByDistance))
	for i, lpoint := range pointsByDistance {
		sortedLabels[i] = lpoint.label
	}

	label := MajorityVote(sortedLabels[:k])

	var classifiedPoint LabeledPoint
	classifiedPoint.point = point
	classifiedPoint.label = label

	return classifiedPoint
}
