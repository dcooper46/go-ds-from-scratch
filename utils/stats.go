package utils

import "math"

// Covariance between two float64 vectors
func Covariance(x, y []float64) float64 {
	mux := Mean(x)
	muy := Mean(y)
	var cov float64
	for i, xi := range x {
		cov += (xi - mux) * (y[i] - muy)
	}
	return cov / float64(len(x)-1)
}

// Correlation gives the linear dependence between
// two float64 vectors
func Correlation(x, y []float64) float64 {
	stdx := StandardDeviation(x)
	stdy := StandardDeviation(y)
	if (stdx > 0.0) && (stdy > 0.0) {
		return Covariance(x, y) / stdx / stdy
	}
	return 0.0

}

// Mean gives the average number of a float64 vector
func Mean(x []float64) float64 {
	var mu float64
	for _, xi := range x {
		mu += xi
	}
	return mu / float64(len(x))
}

// Variance gives how distributed a vector is
// around its mean
func Variance(x []float64) float64 {
	mu := Mean(x)
	var v float64
	for _, xi := range x {
		v += math.Pow(xi-mu, 2)
	}
	return v / float64(len(x)-1)
}

// StandardDeviation gives the unitless dispersion of
// a vector from its mean
func StandardDeviation(x []float64) float64 {
	return math.Sqrt(Variance(x))
}
