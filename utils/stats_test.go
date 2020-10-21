package utils

import (
	"math"
	"testing"
)

func TestMean(t *testing.T) {
	actual := Mean([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
	if actual != 3.0 {
		t.Fatalf("Mean([1, 2, 3, 4, 5]) = %f; want 3.0", actual)
	}
}

func TestVariance(t *testing.T) {
	actual := Variance([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
	if actual != 2.5 {
		t.Fatalf("Variance([1, 2, 3, 4, 5]) = %f; want 2.5", actual)
	}
}

func TestStandardDeviation(t *testing.T) {
	actual := StandardDeviation([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
	roundedActual := math.Round(actual*1000) / 1000
	if roundedActual != 1.581 {
		t.Fatalf("Variance([1, 2, 3, 4, 5]) = %f; want 1.581", roundedActual)
	}
}
