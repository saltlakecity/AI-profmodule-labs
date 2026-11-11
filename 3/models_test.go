package main

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func TestLinearKnownLineAndConstantFeature(t *testing.T) {
	data := []Row{{[]float64{1}, 5}, {[]float64{2}, 7}, {[]float64{3}, 9}, {[]float64{4}, 11}}
	m, loss, err := TrainLinear(data, 0.1, 300)
	if err != nil || math.Abs(m.Predict(10)-23) > 1e-7 || loss[len(loss)-1] >= loss[0] {
		t.Fatalf("%+v %v", m, err)
	}
	m, _, err = TrainLinear([]Row{{[]float64{2}, 4}, {[]float64{2}, 6}}, 0.1, 100)
	if err != nil || math.Abs(m.Predict(2)-5) > 1e-7 {
		t.Fatalf("постоянный признак: %+v %v", m, err)
	}
}

func TestLogisticAndStableSigmoid(t *testing.T) {
	data := []Row{{[]float64{-3, -2}, 0}, {[]float64{-2, -1}, 0}, {[]float64{2, 1}, 1}, {[]float64{3, 2}, 1}}
	m, loss, err := TrainLogistic(data, 0.1, 400)
	if err != nil {
		t.Fatal(err)
	}
	if loss[len(loss)-1] >= loss[0] {
		t.Fatal("ошибка не уменьшается")
	}
	for _, row := range data {
		p, y, err := m.Predict(row.X)
		if err != nil || y != int(row.Y) || p < 0 || p > 1 {
			t.Fatalf("%v: %g %d %v", row, p, y, err)
		}
	}
	if sigmoid(1000) != 1 || sigmoid(-1000) != 0 {
		t.Fatal("неустойчивая sigmoid")
	}
	if _, _, err := m.Predict([]float64{1}); err == nil {
		t.Fatal("размерность")
	}
}

func TestKNNNeighborsTiesAndValidation(t *testing.T) {
	data := []Row{{[]float64{2.4, 3}, 1}, {[]float64{2.7, 3.2}, 1}, {[]float64{1.8, 2.9}, 0}}
	m, err := NewKNN(data, 3)
	if err != nil {
		t.Fatal(err)
	}
	y, neighbors, err := m.Predict([]float64{2.5, 3.1})
	if err != nil || y != 1 || len(neighbors) != 3 || math.Abs(neighbors[0].Distance-math.Sqrt(0.02)) > 1e-12 {
		t.Fatalf("%d %+v %v", y, neighbors, err)
	}
	m, _ = NewKNN([]Row{{[]float64{0}, 0}, {[]float64{2}, 1}}, 2)
	y, _, _ = m.Predict([]float64{1.8})
	if y != 1 {
		t.Fatal("при равенстве голосов выигрывает меньшая сумма расстояний")
	}
	y, _, _ = m.Predict([]float64{1})
	if y != 0 {
		t.Fatal("полное равенство разрешается в пользу класса 0")
	}
	if _, err := NewKNN(data, 4); err == nil {
		t.Fatal("K > N")
	}
	if _, _, err := m.Predict([]float64{math.NaN()}); err == nil {
		t.Fatal("NaN")
	}
}

func TestMetrics(t *testing.T) {
	m := classificationMetrics([]int{0, 0, 1, 1}, []int{0, 1, 0, 1})
	if m.Confusion != [2][2]int{{1, 1}, {1, 1}} || m.Accuracy != 0.5 || m.Precision != 0.5 || m.Recall != 0.5 || m.F1 != 0.5 {
		t.Fatalf("%+v", m)
	}
	m = classificationMetrics([]int{0, 0}, []int{0, 0})
	if m.Accuracy != 1 || m.Precision != 0 || m.Recall != 0 || m.F1 != 0 {
		t.Fatalf("деление на ноль: %+v", m)
	}
	mae, mse, rmse := regressionMetrics([]float64{1, 3}, []float64{2, 1})
	if mae != 1.5 || mse != 2.5 || rmse != math.Sqrt(2.5) {
		t.Fatal(mae, mse, rmse)
	}
}

func TestLeaveOneOutExcludesTheQuery(t *testing.T) {
	data := []Row{{[]float64{0}, 0}, {[]float64{1}, 1}, {[]float64{2}, 0}}
	m, err := leaveOneOut(data, 1)
	if err != nil || m.Accuracy != 0 {
		t.Fatalf("исключённый объект не должен находить сам себя: %+v, %v", m, err)
	}
}

func TestDataAndSplit(t *testing.T) {
	for _, fixture := range []struct {
		content []byte
		target  string
		count   int
	}{{linearCSV, "price", 30}, {logisticCSV, "class", 100}, {knnCSV, "class", 30}} {
		data, err := readData(bytes.NewReader(fixture.content), fixture.target)
		if err != nil || len(data) != fixture.count {
			t.Fatalf("%d %v", len(data), err)
		}
		train, test, err := split(data, 42)
		if err != nil || len(train)+len(test) != len(data) {
			t.Fatal(err)
		}
		for _, a := range train {
			for _, b := range test {
				if &a.X[0] == &b.X[0] {
					t.Fatal("пересечение train и test")
				}
			}
		}
	}
	for _, input := range []string{"x,class\nNaN,0\n", "x,class\n1,2\n", "x,class\n1\n", "x,class\n"} {
		if _, err := readData(strings.NewReader(input), "class"); err == nil {
			t.Fatal("принят некорректный CSV", input)
		}
	}
	if _, _, err := TrainLinear([]Row{{[]float64{1}, 2}}, math.NaN(), 10); err == nil {
		t.Fatal("NaN rate")
	}
	if _, _, err := TrainLogistic([]Row{{[]float64{1}, 2}}, 0.1, 10); err == nil {
		t.Fatal("некорректный класс")
	}
}
