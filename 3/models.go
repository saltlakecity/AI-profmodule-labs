package main

import (
	"fmt"
	"math"
	"sort"
)

type LinearRegression struct{ Weight, Bias float64 }

func TrainLinear(data []Row, rate float64, epochs int) (LinearRegression, []float64, error) {
	model := LinearRegression{}
	if err := validate(data, false); err != nil {
		return model, nil, err
	}
	if len(data[0].X) != 1 || !finite(rate) || rate <= 0 || epochs <= 0 {
		return model, nil, fmt.Errorf("линейная модель требует один признак, rate > 0 и epochs > 0")
	}
	scaler, err := fitScaler(data)
	if err != nil {
		return model, nil, err
	}
	meanY := 0.0
	for _, row := range data {
		meanY += row.Y / float64(len(data))
	}
	scaleY := 0.0
	for _, row := range data {
		scaleY += (row.Y - meanY) * (row.Y - meanY) / float64(len(data))
	}
	scaleY = math.Sqrt(scaleY)
	if !finite(meanY) || !finite(scaleY) {
		return model, nil, fmt.Errorf("переполнение среднего или масштаба целевых значений при нормализации")
	}
	if scaleY == 0 {
		scaleY = 1
	}
	w, b := 0.0, 0.0
	history := make([]float64, 0, epochs)
	for epoch := 0; epoch < epochs; epoch++ {
		dw, db := 0.0, 0.0
		for _, row := range data {
			x := (row.X[0] - scaler.Mean[0]) / scaler.Scale[0]
			y := (row.Y - meanY) / scaleY
			delta := w*x + b - y
			dw += 2 * delta * x / float64(len(data))
			db += 2 * delta / float64(len(data))
		}
		w -= rate * dw
		b -= rate * db
		model.Weight = w * scaleY / scaler.Scale[0]
		model.Bias = meanY + scaleY*b - model.Weight*scaler.Mean[0]
		loss := 0.0
		for _, row := range data {
			d := model.Predict(row.X[0]) - row.Y
			loss += d * d / float64(len(data))
		}
		if !finite(loss) {
			return model, history, fmt.Errorf("линейная регрессия расходится")
		}
		history = append(history, loss)
	}
	return model, history, nil
}

func (m LinearRegression) Predict(x float64) float64 { return m.Weight*x + m.Bias }

type LogisticRegression struct {
	Weights []float64
	Bias    float64
	Scaler  Scaler
}

func sigmoid(z float64) float64 {
	if z >= 0 {
		return 1 / (1 + math.Exp(-z))
	}
	e := math.Exp(z)
	return e / (1 + e)
}

func TrainLogistic(data []Row, rate float64, epochs int) (LogisticRegression, []float64, error) {
	m := LogisticRegression{}
	if err := validate(data, true); err != nil {
		return m, nil, err
	}
	if !finite(rate) || rate <= 0 || epochs <= 0 {
		return m, nil, fmt.Errorf("нужны rate > 0 и epochs > 0")
	}
	scaler, err := fitScaler(data)
	if err != nil {
		return m, nil, err
	}
	m.Weights = make([]float64, len(data[0].X))
	m.Scaler = scaler
	standardized := make([][]float64, len(data))
	for i, row := range data {
		standardized[i], err = m.Scaler.transform(row.X)
		if err != nil {
			return m, nil, err
		}
	}
	history := make([]float64, 0, epochs)
	for epoch := 0; epoch < epochs; epoch++ {
		dw := make([]float64, len(m.Weights))
		db := 0.0
		for i, row := range data {
			z := m.Bias
			for j, x := range standardized[i] {
				z += m.Weights[j] * x
			}
			delta := sigmoid(z) - row.Y
			for j, x := range standardized[i] {
				dw[j] += delta * x / float64(len(data))
			}
			db += delta / float64(len(data))
		}
		for j := range m.Weights {
			m.Weights[j] -= rate * dw[j]
		}
		m.Bias -= rate * db
		loss := 0.0
		for i, row := range data {
			z := m.Bias
			for j, x := range standardized[i] {
				z += m.Weights[j] * x
			}
			loss += (max(z, 0) - row.Y*z + math.Log1p(math.Exp(-math.Abs(z)))) / float64(len(data))
		}
		if !finite(loss) {
			return m, history, fmt.Errorf("логистическая регрессия расходится")
		}
		history = append(history, loss)
	}
	return m, history, nil
}

func (m LogisticRegression) Predict(x []float64) (float64, int, error) {
	values, err := m.Scaler.transform(x)
	if err != nil {
		return 0, 0, err
	}
	z := m.Bias
	for j, value := range values {
		z += m.Weights[j] * value
	}
	p := sigmoid(z)
	if !finite(p) {
		return 0, 0, fmt.Errorf("неконечная вероятность")
	}
	class := 0
	if p >= 0.5 {
		class = 1
	}
	return p, class, nil
}

type Neighbor struct {
	Row
	Distance float64
}
type KNN struct {
	Data []Row
	K    int
}

func NewKNN(data []Row, k int) (KNN, error) {
	if err := validate(data, true); err != nil {
		return KNN{}, err
	}
	if k < 1 || k > len(data) {
		return KNN{}, fmt.Errorf("K должен быть от 1 до %d", len(data))
	}
	return KNN{data, k}, nil
}

func (m KNN) Predict(x []float64) (int, []Neighbor, error) {
	if len(m.Data) == 0 || m.K < 1 || m.K > len(m.Data) {
		return 0, nil, fmt.Errorf("KNN не инициализирован")
	}
	if len(x) != len(m.Data[0].X) {
		return 0, nil, fmt.Errorf("ожидается %d признаков", len(m.Data[0].X))
	}
	for _, value := range x {
		if !finite(value) {
			return 0, nil, fmt.Errorf("неконечный признак")
		}
	}
	neighbors := make([]Neighbor, len(m.Data))
	for i, row := range m.Data {
		d := 0.0
		for j, value := range row.X {
			d = math.Hypot(d, value-x[j])
		}
		if !finite(d) {
			return 0, nil, fmt.Errorf("переполнение расстояния")
		}
		neighbors[i] = Neighbor{row, d}
	}
	sort.SliceStable(neighbors, func(i, j int) bool { return neighbors[i].Distance < neighbors[j].Distance })
	neighbors = neighbors[:m.K]
	votes := [2]int{}
	distances := [2]float64{}
	for _, n := range neighbors {
		class := int(n.Y)
		votes[class]++
		distances[class] += n.Distance
	}
	class := 0
	if votes[1] > votes[0] || (votes[1] == votes[0] && distances[1] < distances[0]) {
		class = 1
	}
	return class, neighbors, nil
}

type ClassificationMetrics struct {
	Confusion                       [2][2]int
	Accuracy, Precision, Recall, F1 float64
}

func leaveOneOut(data []Row, k int) (ClassificationMetrics, error) {
	actual, predicted := make([]int, len(data)), make([]int, len(data))
	for i, row := range data {
		train := make([]Row, 0, len(data)-1)
		train = append(train, data[:i]...)
		train = append(train, data[i+1:]...)
		model, err := NewKNN(train, k)
		if err != nil {
			return ClassificationMetrics{}, err
		}
		class, _, err := model.Predict(row.X)
		if err != nil {
			return ClassificationMetrics{}, err
		}
		actual[i], predicted[i] = int(row.Y), class
	}
	return classificationMetrics(actual, predicted), nil
}

func classificationMetrics(actual, predicted []int) ClassificationMetrics {
	m := ClassificationMetrics{}
	for i, y := range actual {
		m.Confusion[y][predicted[i]]++
	}
	tn, fp, fn, tp := m.Confusion[0][0], m.Confusion[0][1], m.Confusion[1][0], m.Confusion[1][1]
	if len(actual) > 0 {
		m.Accuracy = float64(tn+tp) / float64(len(actual))
	}
	if tp+fp > 0 {
		m.Precision = float64(tp) / float64(tp+fp)
	}
	if tp+fn > 0 {
		m.Recall = float64(tp) / float64(tp+fn)
	}
	if m.Precision+m.Recall > 0 {
		m.F1 = 2 * m.Precision * m.Recall / (m.Precision + m.Recall)
	}
	return m
}

func regressionMetrics(actual, predicted []float64) (mae, mse, rmse float64) {
	for i, y := range actual {
		d := predicted[i] - y
		mae += math.Abs(d) / float64(len(actual))
		mse += d * d / float64(len(actual))
	}
	return mae, mse, math.Sqrt(mse)
}
