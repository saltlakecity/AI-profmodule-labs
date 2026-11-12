package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
)

type Row struct {
	X []float64
	Y float64
}

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

func validate(data []Row, classes bool) error {
	if len(data) == 0 || len(data[0].X) == 0 {
		return fmt.Errorf("пустая выборка")
	}
	for i, row := range data {
		if len(row.X) != len(data[0].X) || !finite(row.Y) {
			return fmt.Errorf("объект %d: некорректная размерность или целевое значение", i+1)
		}
		if classes && row.Y != 0 && row.Y != 1 {
			return fmt.Errorf("объект %d: класс должен быть 0 или 1", i+1)
		}
		for _, x := range row.X {
			if !finite(x) {
				return fmt.Errorf("объект %d: неконечный признак", i+1)
			}
		}
	}
	return nil
}

func readData(r io.Reader, target string) ([]Row, error) {
	reader := csv.NewReader(r)
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("заголовок: %w", err)
	}
	if len(header) < 2 || header[len(header)-1] != target {
		return nil, fmt.Errorf("ожидаются признаки и последний столбец %s", target)
	}
	var data []Row
	for line := 2; ; line++ {
		cells, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("строка %d: %w", line, err)
		}
		row := Row{X: make([]float64, len(header)-1)}
		for j, cell := range cells {
			x, err := strconv.ParseFloat(strings.TrimSpace(cell), 64)
			if err != nil || !finite(x) {
				return nil, fmt.Errorf("строка %d, столбец %d: ожидается конечное число", line, j+1)
			}
			if j < len(row.X) {
				row.X[j] = x
			} else {
				row.Y = x
			}
		}
		data = append(data, row)
	}
	if err := validate(data, target == "class"); err != nil {
		return nil, err
	}
	return data, nil
}

func loadData(path string, embedded []byte, target string) ([]Row, error) {
	if path == "" {
		return readData(bytes.NewReader(embedded), target)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readData(f, target)
}

func split(data []Row, seed uint64) (train, test []Row, err error) {
	if len(data) < 5 {
		return nil, nil, fmt.Errorf("для разбиения 80/20 нужно хотя бы 5 объектов")
	}
	indices := rand.New(rand.NewPCG(seed, seed+1)).Perm(len(data))
	n := len(data) * 4 / 5
	for i, index := range indices {
		if i < n {
			train = append(train, data[index])
		} else {
			test = append(test, data[index])
		}
	}
	return train, test, nil
}

type Scaler struct{ Mean, Scale []float64 }

func fitScaler(data []Row) (Scaler, error) {
	s := Scaler{make([]float64, len(data[0].X)), make([]float64, len(data[0].X))}
	for _, row := range data {
		for j, x := range row.X {
			s.Mean[j] += x / float64(len(data))
		}
	}
	for _, row := range data {
		for j, x := range row.X {
			d := x - s.Mean[j]
			s.Scale[j] += d * d / float64(len(data))
		}
	}
	for j := range s.Scale {
		s.Scale[j] = math.Sqrt(s.Scale[j])
		if !finite(s.Mean[j]) || !finite(s.Scale[j]) {
			return Scaler{}, fmt.Errorf("переполнение среднего или масштаба признака %d при нормализации", j+1)
		}
		if s.Scale[j] == 0 {
			s.Scale[j] = 1
		}
	}
	return s, nil
}

func (s Scaler) transform(x []float64) ([]float64, error) {
	if len(x) != len(s.Mean) {
		return nil, fmt.Errorf("ожидается %d признаков, получено %d", len(s.Mean), len(x))
	}
	z := make([]float64, len(x))
	for j, value := range x {
		if !finite(value) {
			return nil, fmt.Errorf("неконечный признак")
		}
		z[j] = (value - s.Mean[j]) / s.Scale[j]
		if !finite(z[j]) {
			return nil, fmt.Errorf("переполнение при нормализации признака %d", j+1)
		}
	}
	return z, nil
}
