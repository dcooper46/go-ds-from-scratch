package main

import (
	"log"
	"math"
	"math/rand"

	"github.com/dcooper46/go-ds-from-scratch/utils"
)

// Logistic squashes real values between [0, 1]
func Logistic(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// LogisticPrime is the derivative of the logistic function
func LogisticPrime(x float64) float64 {
	return Logistic(x) * (1.0 - Logistic(x))
}

// LogisticLogLikelihoodX returns the log-likelihood of the logistic function
// given an individual record
func LogisticLogLikelihoodX(x []float64, y float64, beta []float64) float64 {
	dot, err := utils.Dot(x, beta)
	if err != nil {
		log.Fatalf("error running dot product: %e", err)
	}
	if y == 1.0 {
		return math.Log(Logistic(dot))
	}
	return math.Log(1.0 - Logistic(dot))
}

// LogisticLogLikelihood returns the log-likelihood of the logistic function
// for a given set of data
func LogisticLogLikelihood(x [][]float64, y []float64, beta []float64) (logLikelihood float64) {
	for i, xi := range x {
		logLikelihood += LogisticLogLikelihoodX(xi, y[i], beta)
	}
	return
}

// LogisticLogGradientX returns the log-likelihood of the logistic function
// for a given record
func LogisticLogGradientX(x []float64, y float64, beta []float64) []float64 {
	parts := make([]float64, len(beta))
	for i := range beta {
		dot, err := utils.Dot(x, beta)
		if err != nil {
			log.Fatalf("error running dot product: %e", err)
		}
		parts[i] = (y - Logistic(dot)) * x[i]
	}
	return parts
}

// LogisticLogGradient returns the gradient of the lostic log likelihood
func LogisticLogGradient(x [][]float64, y, beta []float64) []float64 {
	var logLogGrad []float64
	for i, xi := range x {
		partialGrad := func(a []float64, b float64) []float64 {
			parts := make([]float64, len(beta))
			for i := range beta {
				dot, err := utils.Dot(a, beta)
				if err != nil {
					log.Fatalf("error running dot product: %e", err)
				}
				parts[i] = (b - Logistic(dot)) * a[i]
			}
			return parts
		}(xi, y[i])
		newLogLogGrad, err := utils.VectorAdd(logLogGrad, partialGrad)
		if err != nil {
			log.Fatalf("error adding vectors: %e", err)
		}
		logLogGrad = newLogLogGrad
	}
	return logLogGrad
}

// EstimateBeta uses stochastic gradient decent to find
// coefficents that minimize the squared loss
func EstimateBeta(x [][]float64, y []float64) []float64 {
	betaInit := make([]float64, len(x[0]))
	for i := range betaInit {
		betaInit[i] = rand.Float64()
	}
	return utils.StochasticGradientAscent(
		LogisticLogLikelihoodX,
		LogisticLogGradientX,
		x,
		y,
		betaInit,
		0.001,
		10,
	)
}
