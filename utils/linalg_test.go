package utils

import (
	"math"
	"testing"
)

func TestVectorEqual(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	y1 := []float64{2.0, 3.0, 4.0, 5.0, 6.0}
	y2 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}

	xEqY1 := VectorsEqual(x, y1)
	xEqY2 := VectorsEqual(x, y2)

	if xEqY1 {
		t.Fatalf(
			"VectorsEqual([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) = %v; want false",
			xEqY1,
		)
	}
	if !xEqY2 {
		t.Fatalf(
			"VectorsEqual([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) = %v; want true",
			xEqY2,
		)
	}
}

func TestDot(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	y := []float64{2.0, 3.0, 4.0, 5.0, 6.0}
	actual, err := Dot(x, y)
	if err != nil {
		t.Fatalf(
			"error calling Dot([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]): %s",
			err,
		)
	}
	roundedActual := math.Round(actual*10) / 10
	if roundedActual != 70.0 {
		t.Fatalf(
			"Dot([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) = %f; want 70.0",
			roundedActual,
		)
	}
}

func TestVectorAdd(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	y := []float64{2.0, 3.0, 4.0, 5.0, 6.0}
	actual, err := VectorAdd(x, y)
	if err != nil {
		t.Fatalf(
			"error calling VectorAdd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]): %s",
			err,
		)
	}
	if !VectorsEqual(actual, []float64{3.0, 5.0, 7.0, 9.0, 11.0}) {
		t.Fatalf(
			"VectorAdd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) = %v; want [3, 5, 7, 9, 11]",
			actual,
		)
	}
}

func TestVectorSub(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	y := []float64{2.0, 3.0, 4.0, 5.0, 6.0}
	actual, err := VectorSub(x, y)
	if err != nil {
		t.Fatalf(
			"error calling VectorSub([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]): %s",
			err,
		)
	}
	if !VectorsEqual(actual, []float64{-1.0, -1.0, -1.0, -1.0, -1.0}) {
		t.Fatalf(
			"VectorSub([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) = %v; want [-1, -1, -1, -1, -1]",
			actual,
		)
	}
}

func TestScalarMultiply(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	actual := ScalarMultiply(2.0, x)
	if !VectorsEqual(actual, []float64{2.0, 4.0, 6.0, 8.0, 10.0}) {
		t.Fatalf(
			"ScalarMultiply(2, [1, 2, 3, 4, 5]) = %v; want [2, 4, 6, 8, 10]",
			actual,
		)
	}
}

func TestTranspose(t *testing.T) {
	mat := [][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
		{2.0, 3.0, 4.0, 5.0, 6.0},
	}
	actual := Transpose(mat)
	expected := [][]float64{
		{1.0, 2.0},
		{2.0, 3.0},
		{3.0, 4.0},
		{4.0, 5.0},
		{5.0, 6.0},
	}
	for i, actualRow := range actual {
		if !VectorsEqual(actualRow, expected[i]) {
			t.Fatalf(
				"Transpose([[1, 2, 3, 4, 5],[2, 3, 4, 5, 6]])[%d] = %v;"+
					"want %v",
				i,
				actualRow,
				expected[i],
			)
		}
	}

}
func TestColumnSums(t *testing.T) {
	mat := [][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
		{2.0, 3.0, 4.0, 5.0, 6.0},
	}
	actual, err := ColumnSums(mat)
	if err != nil {
		t.Fatalf(
			"error calling ColumnSums([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]): %s",
			err,
		)
	}
	if !VectorsEqual(actual, []float64{3.0, 5.0, 7.0, 9.0, 11.0}) {
		t.Fatalf(
			"ColumnSums([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]) = %v; want [3, 5, 7, 9, 11]",
			actual,
		)
	}
}

func TestColumnMeans(t *testing.T) {
	mat := [][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
		{2.0, 3.0, 4.0, 5.0, 6.0},
	}
	actual, err := ColumnMeans(mat)
	if err != nil {
		t.Fatalf(
			"error calling ColumnMeans([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]): %s",
			err,
		)
	}
	if !VectorsEqual(actual, []float64{1.5, 2.5, 3.5, 4.5, 5.5}) {
		t.Fatalf(
			"ColumnMeans([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]) = %v; want [1.5, 2.5, 3.5, 4.5, 5.5]",
			actual,
		)
	}
}

func TestMatMult(t *testing.T) {
	A := [][]float64{
		{1.0, 2.0, 3.0, 4.0, 5.0},
		{2.0, 3.0, 4.0, 5.0, 6.0},
	}
	B := [][]float64{
		{3.0, 4.0},
		{5.0, 6.0},
		{7.0, 4.0},
		{5.0, 6.0},
		{7.0, 8.0},
	}
	actual, err := MatMult(A, B)
	if err != nil {
		t.Fatalf(
			"error calling MatMult([[1, 2, 3, 4, 5],[2, 3, 4, 5, 6]], "+
				"[[3, 4], [5, 6], [7, 4], [5, 6], [7, 8]]): %s",
			err,
		)
	}
	expected := [][]float64{
		{89.0, 92.0},
		{116.0, 120.0},
	}
	for i, actualRow := range actual {
		if !VectorsEqual(actualRow, expected[i]) {
			t.Fatalf(
				"MatMult([[1, 2, 3, 4, 5],[2, 3, 4, 5, 6]], "+
					"[[3, 4], [5, 6], [7, 4], [5, 6], [7, 8]])[%d] = %v;"+
					"want %v",
				i,
				actualRow,
				expected[i],
			)
		}
	}
	_, errt := MatMult(A, Transpose(B))
	if errt == nil {
		t.Fatalf(
			"MatMult([[1, 2, 3, 4, 5],[2, 3, 4, 5, 6]], "+
				"[[3, 5, 7, 5, 7], [4, 6, 4, 6, 8]])"+
				"should raise error: "+
				"incompatible shapes: (%d,%d), (%d,%d)",
			2, 5, 2, 5,
		)
	}
}
