package main

import (
	"log"
	"math"
	"math/rand"

	"github.com/dcooper46/go-ds-from-scratch/utils"
)

// Predict estimates a regression value given inputs and coefficients
// assumes first element is 1
func Predict(x, beta []float64) (float64, error) {
	return utils.Dot(x, beta)
}

// Error gives the difference between a prediction and the expected value
func Error(x []float64, y float64, beta []float64) float64 {
	pred, err := Predict(x, beta)
	if err != nil {
		log.Fatalf("prediction error: %e", err)
	}
	return y - pred
}

// SquaredError gives the squared difference between
// a prediction and the expected value
func SquaredError(x []float64, y float64, beta []float64) float64 {
	return math.Pow(Error(x, y, beta), 2)
}

// TotalSumOfSquares gives the unnormalized variace
func TotalSumOfSquares(y []float64) (tss float64) {
	muy := utils.Mean(y)
	for _, yi := range y {
		tss += math.Pow(yi-muy, 2)
	}
	return
}

// SquaredErrorGradient gives the gradient for the squared error function
func SquaredErrorGradient(x []float64, y float64, beta []float64) []float64 {
	grad := make([]float64, len(x))
	for i, xi := range x {
		grad[i] = -2 * xi * Error(x, y, beta)
	}
	return grad
}

// EstimateBeta uses stochastic gradient decent to find
// coefficents that minimize the squared loss
func EstimateBeta(x [][]float64, y []float64) []float64 {
	betaInit := make([]float64, len(x[0]))
	for i := range betaInit {
		betaInit[i] = rand.Float64()
	}
	return utils.StochasticGradientDecent(
		SquaredError,
		SquaredErrorGradient,
		x,
		dailyMins,
		betaInit,
		0.001,
		10,
	)
}

// RSquared gives the variance in y explained by the model
func RSquared(x [][]float64, y []float64, beta []float64) float64 {
	var sse float64
	for i, xi := range x {
		sse += SquaredError(xi, y[i], beta)
	}
	return 1.0 - sse/TotalSumOfSquares(y)
}

// RidgePenalty add a penalty proportional to the sum of squares
func RidgePenalty(beta []float64, alpha float64) float64 {
	dot, err := utils.Dot(beta[1:], beta[1:])
	if err != nil {
		log.Fatalf("error performing dot product: %e", err)
	}
	return alpha * dot
}

// SquaredErrorRidge estimates the squared error plus ridge penalty on beta
func SquaredErrorRidge(
	x []float64,
	y float64,
	beta []float64,
	alpha float64,
) float64 {
	return SquaredError(x, y, beta) + RidgePenalty(beta, alpha)
}

// SquaredErrorRidgeAlpha gives the gradient of the ith squared error term
func SquaredErrorRidgeAlpha(
	alpha float64,
) func([]float64, float64, []float64) float64 {
	return func(x []float64, y float64, beta []float64) float64 {
		return SquaredErrorRidge(x, y, beta, alpha)
	}
}

// RidgeGradient gives the gradient of the ridge penalty
func RidgeGradient(beta []float64, alpha float64) []float64 {
	grad := make([]float64, len(beta))
	grad[0] = 0.0
	for i, b := range beta[1:] {
		grad[i+1] = 2 * alpha * b
	}
	return grad
}

// SquaredErrorRidgeGradient gives the gradient of the ith squared error term
func SquaredErrorRidgeGradient(
	x []float64,
	y float64,
	beta []float64,
	alpha float64,
) []float64 {
	vsum, err := utils.VectorAdd(
		SquaredErrorGradient(x, y, beta),
		RidgeGradient(beta, alpha),
	)
	if err != nil {
		log.Fatalf("error adding gradients: %e", err)
	}
	return vsum
}

// SquaredErrorRidgeGradientAlpha gives the gradient of the ith squared error term
func SquaredErrorRidgeGradientAlpha(
	alpha float64,
) func([]float64, float64, []float64) []float64 {
	return func(x []float64, y float64, beta []float64) []float64 {
		return SquaredErrorRidgeGradient(x, y, beta, alpha)
	}
}

// EstimateBetaRidge uses gradient decent to fit parameters
// with a ridge penalty
func EstimateBetaRidge(
	x [][]float64,
	y []float64,
	alpha float64,
) []float64 {
	betaInit := make([]float64, len(x[0]))
	for i := range betaInit {
		betaInit[i] = rand.Float64()
	}
	return utils.StochasticGradientDecent(
		SquaredErrorRidgeAlpha(alpha),
		SquaredErrorRidgeGradientAlpha(alpha),
		x,
		dailyMins,
		betaInit,
		0.001,
		50,
	)
}
