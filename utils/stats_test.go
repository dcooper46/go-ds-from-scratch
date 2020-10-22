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

func TestCovariance(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	y := []float64{1.0, 2.1, 3.5, 4.4, 5.3}
	actual := Covariance(x, y)
	roundedActual := math.Round(actual*1000) / 1000
	if roundedActual != 2.725 {
		t.Fatalf(
			"Covariance([1, 2, 3, 4, 5], [1, 2.1, 3.5, 4.4, 5.3]) = %f; want 2.725",
			roundedActual,
		)
	}
}

func TestCorrelation(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	y := []float64{1.0, 2.1, 3.5, 4.4, 5.3}
	actual := Correlation(x, y)
	roundedActual := math.Round(actual*1000) / 1000
	if roundedActual != 0.996 {
		t.Fatalf(
			"Correlation([1, 2, 3, 4, 5], [1, 2.1, 3.5, 4.4, 5.3]) = %f; want 0.996",
			roundedActual,
		)
	}
}
