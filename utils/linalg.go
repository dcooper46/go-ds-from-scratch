package utils

import (
	"fmt"
)

// VectorsEqual determine if two vectors are equal in their elements
func VectorsEqual(x, y []float64) bool {
	if len(x) != len(y) {
		return false
	}
	for i, xi := range x {
		if xi != y[i] {
			return false
		}
	}
	return true
}

// Dot returns the dot or inner product between vectors
func Dot(x, y []float64) (float64, error) {
	if len(x) != len(y) {
		return 0.0, fmt.Errorf("vectors are of unequal size: %d != %d", len(x), len(y))
	}
	var dot float64
	for i, xi := range x {
		dot += xi * y[i]
	}
	return dot, nil
}

// VectorAdd adds two vectors elementwise
func VectorAdd(x, y []float64) ([]float64, error) {
	if len(x) != len(y) {
		return []float64{}, fmt.Errorf("vectors are of unequal size: %d != %d", len(x), len(y))
	}
	vsum := make([]float64, len(x))
	for i, xi := range x {
		vsum[i] = xi + y[i]
	}
	return vsum, nil
}

// VectorSub subtracts two vectors elementwise
func VectorSub(x, y []float64) ([]float64, error) {
	if len(x) != len(y) {
		return []float64{}, fmt.Errorf("vectors are of unequal size: %d != %d", len(x), len(y))
	}
	vsub := make([]float64, len(x))
	for i, xi := range x {
		vsub[i] = xi - y[i]
	}
	return vsub, nil
}

// ScalarMultiply multiplies each element of a vector by a scalar
func ScalarMultiply(s float64, vec []float64) []float64 {
	svec := make([]float64, len(vec))
	for i, v := range vec {
		svec[i] = s * v
	}
	return svec
}

// Shape returns the dimensions of a nested slice (matrix)
func Shape(mat [][]float64) (nrow, ncol int) {
	return len(mat), len(mat[0])
}

// Transpose flips a matrix along its axes
func Transpose(mat [][]float64) [][]float64 {
	nrow, ncol := Shape(mat)
	tmat := make([][]float64, ncol)
	for i := 0; i < ncol; i++ {
		tmat[i] = make([]float64, nrow)
		for j := 0; j < nrow; j++ {
			tmat[i][j] = mat[j][i]
		}
	}
	return tmat
}

// ColumnSums sums elements of a nested slice (matrix) along the first axis
func ColumnSums(mat [][]float64) ([]float64, error) {
	colsums := make([]float64, len(mat[0]))
	for _, row := range mat {
		for j, col := range row {
			if j > len(colsums) {
				return []float64{}, fmt.Errorf("ValueError: bad vector %v", row)
			}
			colsums[j] += col
		}
	}
	return colsums, nil
}

// ColumnMeans averages elements of a nested slice (matrix) along the first axis
func ColumnMeans(mat [][]float64) ([]float64, error) {
	colsums, err := ColumnSums(mat)
	if err != nil {
		return []float64{}, err
	}
	n := float64(len(mat))
	return ScalarMultiply(1.0/n, colsums), nil
}

// MatMult performs matrix multiplication
func MatMult(a, b [][]float64) ([][]float64, error) {
	nrowa, ncola := Shape(a)
	nrowb, ncolb := Shape(b)
	if ncola != nrowb {
		return nil, fmt.Errorf(
			"incompatible shapes: (%d,%d), (%d,%d)",
			nrowa, ncola, nrowb, ncolb)
	}
	returnMat := make([][]float64, nrowa)
	for i := 0; i < nrowa; i++ {
		returnMat[i] = make([]float64, ncolb)
		for j := 0; j < ncolb; j++ {
			for k := 0; k < ncola; k++ {
				returnMat[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return returnMat, nil
}
