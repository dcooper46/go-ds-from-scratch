package utils

import (
	"math/rand"
)

// TrainTestSplit splits a data set into groups
// for training and testing a model
func TrainTestSplit(
	x [][]float64,
	y []float64,
	testP float64,
) ([][]float64, [][]float64, []float64, []float64) {
	var xTrain, xTest [][]float64
	var yTrain, yTest []float64
	for i, row := range x {
		if rand.Float64() < testP {
			xTest = append(xTest, row)
			yTest = append(yTest, y[i])
		} else {
			xTrain = append(xTrain, row)
			yTrain = append(yTrain, y[i])
		}
	}
	return xTrain, xTest, yTrain, yTest
}

// Confusion returns a confusion matrix of predictions
// assume binary labels
// rows are true labels and columns are predicted labels
func Confusion(y, pred []float64) [][]float64 {
	confMat := make([][]float64, 2)
	confMat[0] = make([]float64, 2)
	confMat[1] = make([]float64, 2)

	for i, yi := range y {
		if yi == 1.0 {
			if pred[i] == 1.0 {
				confMat[1][1]++
			} else {
				confMat[1][0]++
			}
		} else {
			if pred[i] == 0.0 {
				confMat[0][0]++
			} else {
				confMat[0][1]++
			}
		}
	}
	return confMat
}

// Precision accepts a 2x2 confusion matrix and returns
// how precise model predictions are
func Precision(confMat [][]float64) float64 {
	return confMat[1][1] / VectorSum(GetColumn(confMat, 1))
}

// Recall accepts a 2x2 confusion matrix and returns
// how many actual positives were predicted correctly
func Recall(confMat [][]float64) float64 {
	return confMat[1][1] / RowSums(confMat)[1]
}
