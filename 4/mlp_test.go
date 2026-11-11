package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

func TestBackpropagationAgainstFiniteDifferences(t *testing.T) {
	m := NewMLP(2, 3, 7)
	x, target := []float64{0.3, -0.8}, [2]float64{2, -3}
	gradient := make([]float64, len(m.Parameters))
	m.accumulateGradient(x, target, make([]float64, m.Hidden), gradient)
	loss := func() float64 {
		y := m.forward(x, make([]float64, m.Hidden))
		return (math.Abs(y[0]-target[0]) + math.Abs(y[1]-target[1])) / 2
	}
	const eps = 1e-6
	for i := range m.Parameters {
		original := m.Parameters[i]
		m.Parameters[i] = original + eps
		plus := loss()
		m.Parameters[i] = original - eps
		minus := loss()
		m.Parameters[i] = original
		numerical := (plus - minus) / (2 * eps)
		if math.Abs(gradient[i]-numerical) > 1e-7 {
			t.Fatalf("параметр %d: backprop = %g, численный = %g", i, gradient[i], numerical)
		}
	}
}

func TestMLPLearnsTwoOutputsAndIsReproducible(t *testing.T) {
	var data []Example
	for i := 0; i < 25; i++ {
		x := float64(i)/12 - 1
		data = append(data, Example{Input: []float64{x, x * x}, Target: [2]float64{2*x + 1, -x + 0.5}})
	}
	m := NewMLP(2, 8, 42)
	loss, err := TrainMLP(&m, data, 400, 0.01)
	if err != nil {
		t.Fatal(err)
	}
	if loss[len(loss)-1] > 0.07 || loss[len(loss)-1] >= loss[0] {
		t.Fatalf("MLP не обучился: %g -> %g", loss[0], loss[len(loss)-1])
	}
	p := m.forward([]float64{0.25, 0.25 * 0.25}, make([]float64, m.Hidden))
	if math.Abs(p[0]-1.5) > 0.12 || math.Abs(p[1]-0.25) > 0.12 {
		t.Fatalf("прогноз двух выходов: %v", p)
	}
	m2 := NewMLP(2, 8, 42)
	if _, err := TrainMLP(&m2, data, 400, 0.01); err != nil {
		t.Fatal(err)
	}
	for i := range m.Parameters {
		if m.Parameters[i] != m2.Parameters[i] {
			t.Fatal("seed не обеспечивает повторяемость")
		}
	}
	encoded, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	var restored MLP
	if err := json.Unmarshal(encoded, &restored); err != nil {
		t.Fatal(err)
	}
	if restored.forward([]float64{0.25, 0.0625}, make([]float64, restored.Hidden)) != p {
		t.Fatal("сохранённые веса не восстанавливают прогноз")
	}
}

func weatherFixture() string {
	var b strings.Builder
	b.WriteString("date,temperature,precipitation\n")
	start := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	for i := 0; i < 25; i++ {
		fmt.Fprintf(&b, "%s,%d,%d\n", start.AddDate(0, 0, i).Format("2006-01-02"), i, i%3)
	}
	return b.String()
}

func TestWeatherWindowsAndTrainingOnlyNormalization(t *testing.T) {
	days, err := readWeather(strings.NewReader(weatherFixture()))
	if err != nil {
		t.Fatal(err)
	}
	data := makeExamples(days)
	if len(data) != 18 || len(data[0].Input) != 16 || !data[0].Date.Equal(days[7].Date) {
		t.Fatal("неверное окно")
	}
	for i, row := range data {
		if row.Input[0] != float64(i) || row.Input[12] != float64(i+6) || row.Target[0] != float64(i+7) {
			t.Fatalf("сдвиг цели или утечка: %+v", row)
		}
		if row.Previous[0] != row.Input[12] {
			t.Fatal("неверная база «как вчера»")
		}
	}
	trainCount := len(data) * 4 / 5
	_, beforeX, beforeY := normalizeTraining(data[:trainCount])
	for i := trainCount; i < len(data); i++ {
		data[i].Target = [2]float64{1e9, 1e9}
		for j := range data[i].Input {
			data[i].Input[j] = 1e9
		}
	}
	_, afterX, afterY := normalizeTraining(data[:trainCount])
	for j := range beforeX.Mean {
		if beforeX.Mean[j] != afterX.Mean[j] || beforeX.Scale[j] != afterX.Scale[j] {
			t.Fatal("test влияет на нормализацию признаков")
		}
	}
	for j := range beforeY.Mean {
		if beforeY.Mean[j] != afterY.Mean[j] || beforeY.Scale[j] != afterY.Scale[j] {
			t.Fatal("test влияет на нормализацию целей")
		}
	}
	if beforeX.Mean[0] != 6.5 || beforeY.Mean[0] != 13.5 {
		t.Fatalf("неверные параметры train: %v %v", beforeX.Mean, beforeY.Mean)
	}
}

func TestWeatherInputValidationAndRealDataset(t *testing.T) {
	valid := weatherFixture()
	for _, input := range []string{
		"date,temperature,precipitation\n",
		strings.Replace(valid, "2020-01-02,1,1", "2020-01-01,1,1", 1),
		strings.Replace(valid, "2020-01-02,1,1\n", "", 1),
		strings.Replace(valid, "2020-01-02,1,1", "2020-01-02,NaN,1", 1),
		strings.Replace(valid, "2020-01-02,1,1", "2020-01-02,1,-1", 1),
		strings.Replace(valid, "2020-01-02,1,1", "bad-date,1,1", 1),
	} {
		if _, err := readWeather(strings.NewReader(input)); err == nil {
			t.Fatal("принят некорректный CSV")
		}
	}
	days, err := readWeather(bytes.NewReader(weatherCSV))
	if err != nil || len(days) != 16071 {
		t.Fatalf("датасет: %d %v", len(days), err)
	}
	if days[0].Date.Format("2006-01-02") != "1980-01-01" || days[len(days)-1].Date.Format("2006-01-02") != "2023-12-31" {
		t.Fatal("изменился период данных")
	}
}

func TestPredictionUnitsAndInvalidTraining(t *testing.T) {
	m := NewMLP(1, 1, 1)
	clear(m.Parameters)
	// При нулевых весах выход определяется смещением последнего слоя.
	m.Parameters[3] = 0.5
	m.Parameters[5] = -2
	p, err := predictWeather(m, []float64{5}, Scaler{[]float64{0}, []float64{1}}, Scaler{[]float64{10, 1}, []float64{2, 3}})
	if err != nil || p != [2]float64{11, 0} {
		t.Fatalf("единицы / отрицательные осадки: %v %v", p, err)
	}
	if _, err := TrainMLP(&m, nil, 10, 0.01); err == nil {
		t.Fatal("пустая выборка")
	}
	if _, err := TrainMLP(&m, []Example{{Input: []float64{1}, Target: [2]float64{0, 0}}}, 10, math.NaN()); err == nil {
		t.Fatal("NaN rate")
	}
	if _, err := TrainMLP(&m, []Example{{Input: []float64{1, 2}, Target: [2]float64{0, 0}}}, 10, 0.01); err == nil {
		t.Fatal("размерность")
	}
}
