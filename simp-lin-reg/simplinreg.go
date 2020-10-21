package main

import (
	"math"

	"github.com/dcooper46/go-ds-from-scratch/utils"
)

// Predict estimates y given alpha and beta parameters
func Predict(alpha, beta, x float64) float64 {
	return beta*x + alpha
}

// Error gives the difference between the prediction and the truth
func Error(alpha, beta, x, y float64) float64 {
	return y - Predict(alpha, beta, x)
}

// SumOfSquaredErrors returns the sum of predicted errors that have been squared
func SumOfSquaredErrors(alpha, beta float64, x, y []float64) (sse float64) {
	for i, xi := range x {
		sse += math.Pow(Error(alpha, beta, xi, y[i]), 2)
	}
	return
}

// TotalSumOfSquares gives the unnormalized variace
func TotalSumOfSquares(y []float64) (tss float64) {
	muy := utils.Mean(y)
	for _, yi := range y {
		tss += math.Pow(yi-muy, 2)
	}
	return
}

// RSquared returns the coefficient of determination of a fit model
// The coefficient of determination shows how much variance in the
// dependent variable is explained by the model
func RSquared(alpha, beta float64, x, y []float64) (rSqrd float64) {
	rSqrd = 1.0 - SumOfSquaredErrors(alpha, beta, x, y)/TotalSumOfSquares(y)
	return
}

// LeastSquares fits a simple linear model using least squares estimates
// of parameters using mle derivations, given x, y slices of float64
func LeastSquares(x, y []float64) (alpha, beta float64) {
	beta = utils.Correlation(x, y) * utils.StandardDeviation(y) / utils.StandardDeviation(x)
	alpha = utils.Mean(y) - beta*utils.Mean(x)
	return
}
