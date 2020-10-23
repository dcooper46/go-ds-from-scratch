package utils

import "math"

// DifferenceQuotient evaluates a univariate function change at a value and
// a small step above that value.  Taking the limit of this as
// the step approaches 0 yields the derivative of this function
func DifferenceQuotient(f func(xf float64) float64, x, h float64) float64 {
	return (f(x+h) - f(x)) / h
}

// PartialDifferenceQuotient evaluates multivariate function change at each
// variable by treating it as a univariate function, holding the others fixed.
func PartialDifferenceQuotient(
	f func(xf []float64) float64,
	v []float64,
	i int,
	h float64) float64 {
	w := make([]float64, len(v))
	for j := 0; j < len(v); j++ {
		if j == i {
			w[j] = v[j] + h
		} else {
			w[j] = v[j]
		}
	}
	return (f(w) - f(v)) / h
}

// EstimateGradient estimates the gradient of a multivariate function
func EstimateGradient(f func(xf []float64) float64, v []float64, h float64) []float64 {
	grads := make([]float64, len(v))
	for i := 0; i < len(v); i++ {
		grads[i] = PartialDifferenceQuotient(f, v, i, h)
	}
	return grads
}

// Negate returns the oppositive of a given function
func Negate(f func(v []float64) float64) func(v []float64) float64 {
	return func(v []float64) float64 {
		return -f(v)
	}
}

// NegateAll returns the oppositive of a given function
func NegateAll(f func(v []float64) []float64) func(v []float64) []float64 {
	return func(v []float64) []float64 {
		results := f(v)
		ret := make([]float64, len(results))
		for i, r := range results {
			ret[i] = -r
		}
		return ret
	}
}

// Step moves the variables an amount in the direction
// of the gradient
func Step(v, direction []float64, stepSize float64) []float64 {
	directedStep := ScalarMultiply(stepSize, direction)
	moved, _ := VectorAdd(v, directedStep)
	return moved
}

// BatchGradientDecent performs batch gradient decent over the entire dataset
// to find the parameters that minimize a given function
func BatchGradientDecent(
	f func(v []float64) float64,
	g func(v []float64) []float64,
	theta0 []float64,
	tol float64) []float64 {

	stepSizes := []float64{100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001}

	theta := theta0
	value := f(theta)

	nextThetas := make([][]float64, len(stepSizes))

	for {
		gradient := g(theta)
		for i, stepSize := range stepSizes {
			nextThetas[i] = Step(theta, gradient, -1.0*stepSize)
		}
		nextTheta := func(thetas [][]float64) []float64 {
			mt := thetas[0]
			minVal := f(mt)

			for _, t := range thetas[1:] {
				tval := f(t)
				if tval < minVal {
					minVal = tval
					mt = t
				}
			}
			return mt
		}(nextThetas)

		nextValue := f(nextTheta)

		if math.Abs(value-nextValue) < tol {
			return theta
		}

		theta = nextTheta
		value = nextValue

	}
}

// StochasticGradientDecent performs gradient decent on random shuffles of data
// updating one record at a time instead of in batch
func StochasticGradientDecent(
	f func(a []float64, b float64, t []float64) float64,
	g func(a []float64, b float64, t []float64) []float64,
	x [][]float64,
	y, theta0 []float64,
	alpha0 float64,
	maxIter int,
) []float64 {

	theta := theta0
	alpha := alpha0
	minTheta := make([]float64, len(theta))
	minValue := math.Inf(1)
	iterationsNoBetter := 0

	for iterationsNoBetter < maxIter {
		var value float64
		for i, xi := range x {
			value += f(xi, y[i], theta)
		}

		if value < minValue {
			minTheta = theta
			minValue = value
			iterationsNoBetter = 0
			alpha = alpha0
		} else {
			iterationsNoBetter++
			alpha = alpha * 0.9
		}
		for i, xi := range x {
			gradI := g(xi, y[i], theta)
			theta, _ = VectorSub(theta, ScalarMultiply(alpha, gradI))
		}
	}
	return minTheta
}

// StochasticGradientAscent performs gradient ascent on random shuffles of data
// updating one record at a time instead of in batch
func StochasticGradientAscent(
	f func(a []float64, b float64, t []float64) float64,
	g func(a []float64, b float64, t []float64) []float64,
	x [][]float64,
	y, theta0 []float64,
	alpha0 float64,
	maxIter int,
) []float64 {
	return StochasticGradientDecent(
		func(a []float64, b float64, t []float64) float64 {
			return -f(a, b, t)
		},
		func(a []float64, b float64, t []float64) []float64 {
			results := g(a, b, t)
			ret := make([]float64, len(results))
			for i, r := range results {
				ret[i] = -r
			}
			return ret
		},
		x,
		y,
		theta0,
		alpha0,
		maxIter,
	)
}
