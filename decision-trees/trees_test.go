package main

import (
	"math"
	"testing"
)

func TestGetProbabilities(t *testing.T) {
	labels := []string{"a", "b", "a", "c", "a", "b", "b", "c", "c", "b"}
	actual := GetProbabilities(labels)
	expected := []float64{0.30, 0.40, 0.30}
	for i, ai := range actual {
		rai := math.Round(ai*100) / 100
		if rai != expected[i] {
			t.Errorf("wrong probability! got %f, wanted %f", rai, expected[i])
		}
	}
}

func TestEntropy(t *testing.T) {
	probs := []float64{0.3, 0.25, 0.10, 0.35}
	actual := Entropy(probs)
	roundedActual := math.Round(actual*1000) / 1000
	if roundedActual != 1.883 {
		t.Errorf("wrong entropy! got %f, wanted %f", roundedActual, 1.883)
	}
}
