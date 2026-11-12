package main

import (
	"bytes"
	_ "embed"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
)

//go:embed data/train.csv
var trainingCSV []byte

//go:embed data/test.csv
var testingCSV []byte

type Sample struct {
	X []float64
	Y int
}

type Perceptron struct {
	Weights []float64
	Bias    float64
}

func (p Perceptron) Predict(x []float64) (int, error) {
	if len(x) != len(p.Weights) {
		return 0, fmt.Errorf("ожидается %d признаков, получено %d", len(p.Weights), len(x))
	}
	score := p.Bias
	for j, value := range x {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, fmt.Errorf("признак %d не является конечным числом", j+1)
		}
		score += p.Weights[j] * value
	}
	if math.IsNaN(score) || math.IsInf(score, 0) {
		return 0, fmt.Errorf("переполнение при классификации")
	}
	if score >= 0 {
		return 1, nil
	}
	return 0, nil
}

func Train(data []Sample, rate float64, epochs int) (Perceptron, []int, error) {
	if len(data) == 0 || len(data[0].X) == 0 {
		return Perceptron{}, nil, fmt.Errorf("обучающая выборка пуста")
	}
	if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) || epochs <= 0 {
		return Perceptron{}, nil, fmt.Errorf("скорость обучения и число эпох должны быть положительными")
	}
	p := Perceptron{Weights: make([]float64, len(data[0].X))}
	for i, sample := range data {
		if sample.Y != 0 && sample.Y != 1 {
			return p, nil, fmt.Errorf("объект %d: класс должен быть 0 или 1", i+1)
		}
		if _, err := p.Predict(sample.X); err != nil {
			return p, nil, fmt.Errorf("объект %d: %w", i+1, err)
		}
	}
	history := make([]int, 0)
	for epoch := 0; epoch < epochs; epoch++ {
		mistakes := 0
		for _, sample := range data {
			prediction, err := p.Predict(sample.X)
			if err != nil {
				return p, history, err
			}
			delta := rate * float64(sample.Y-prediction)
			if delta == 0 {
				continue
			}
			mistakes++
			for j, x := range sample.X {
				p.Weights[j] += delta * x
				if math.IsNaN(p.Weights[j]) || math.IsInf(p.Weights[j], 0) {
					return p, history, fmt.Errorf("переполнение весов: уменьшите скорость обучения или масштаб признаков")
				}
			}
			p.Bias += delta
		}
		history = append(history, mistakes)
		if mistakes == 0 {
			return p, history, nil
		}
	}
	return p, history, fmt.Errorf("достигнут предел %d эпох; сходимость не подтверждена (данные могут быть линейно неразделимы)", epochs)
}

func readCSV(r io.Reader, labeled bool, expectedFeatures []string) ([]Sample, []string, error) {
	reader := csv.NewReader(r)
	header, err := reader.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("заголовок CSV: %w", err)
	}
	header[0] = strings.TrimPrefix(header[0], "\ufeff")
	features := len(header)
	if labeled {
		features--
		if header[len(header)-1] != "class" {
			return nil, nil, fmt.Errorf("последний столбец должен называться class")
		}
	}
	if features < 1 {
		return nil, nil, fmt.Errorf("нужен хотя бы один признак")
	}
	if expectedFeatures != nil && !slices.Equal(header[:features], expectedFeatures) {
		return nil, nil, fmt.Errorf("имена и порядок признаков должны совпадать с обучением: ожидаются %v, получены %v", expectedFeatures, header[:features])
	}
	var data []Sample
	for line := 2; ; line++ {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, fmt.Errorf("строка %d: %w", line, err)
		}
		sample := Sample{X: make([]float64, features)}
		for j := range sample.X {
			x, err := strconv.ParseFloat(strings.TrimSpace(row[j]), 64)
			if err != nil || math.IsNaN(x) || math.IsInf(x, 0) {
				return nil, nil, fmt.Errorf("строка %d, столбец %d: ожидается конечное число", line, j+1)
			}
			sample.X[j] = x
		}
		if labeled {
			sample.Y, err = strconv.Atoi(strings.TrimSpace(row[features]))
			if err != nil || (sample.Y != 0 && sample.Y != 1) {
				return nil, nil, fmt.Errorf("строка %d: класс должен быть 0 или 1", line)
			}
		}
		data = append(data, sample)
	}
	if len(data) == 0 {
		return nil, nil, fmt.Errorf("CSV не содержит объектов")
	}
	return data, header[:features], nil
}

func loadCSV(path string, fallback []byte, labeled bool, expectedFeatures []string) ([]Sample, []string, error) {
	if path == "" {
		return readCSV(bytes.NewReader(fallback), labeled, expectedFeatures)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	return readCSV(f, labeled, expectedFeatures)
}

func run() error {
	trainPath := flag.String("train", "", "обучающий CSV: признаки, затем class (по умолчанию встроенный пример)")
	testPath := flag.String("test", "", "CSV новых объектов без class, с теми же именами и порядком признаков")
	rate := flag.Float64("rate", 0.1, "скорость обучения")
	epochs := flag.Int("epochs", 1000, "максимум эпох")
	flag.Parse()
	data, features, err := loadCSV(*trainPath, trainingCSV, true, nil)
	if err != nil {
		return err
	}
	test, _, err := loadCSV(*testPath, testingCSV, false, features)
	if err != nil {
		return err
	}
	fmt.Printf("ПЕРСЕПТРОН\nОбъектов: %d; признаков: %d; rate = %g\nИнициализация: нулевые веса и bias\n", len(data), len(data[0].X), *rate)
	model, history, trainErr := Train(data, *rate, *epochs)
	for epoch, count := range history {
		fmt.Printf("Эпоха %d: ошибок = %d\n", epoch+1, count)
	}
	for j, w := range model.Weights {
		fmt.Printf("Вес w%d = %.6f\n", j+1, w)
	}
	fmt.Printf("Bias = %.6f\n", model.Bias)
	if trainErr != nil {
		return trainErr
	}
	fmt.Println("Обучение завершено.\nТест:")
	for _, sample := range test {
		class, err := model.Predict(sample.X)
		if err != nil {
			return err
		}
		fmt.Printf("%v -> класс %d\n", sample.X, class)
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}
}
