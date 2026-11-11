package main

import (
	_ "embed"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
)

//go:embed data/linear.csv
var linearCSV []byte

//go:embed data/logistic.csv
var logisticCSV []byte

//go:embed data/knn.csv
var knnCSV []byte

func printMetrics(m ClassificationMetrics) {
	fmt.Printf("Accuracy = %.2f%%\nPrecision = %.2f%%\nRecall = %.2f%%\nF1-score = %.2f%%\nМатрица ошибок (строки: факт, столбцы: прогноз):\n\tPred 0\tPred 1\nReal 0\t%d\t%d\nReal 1\t%d\t%d\n", 100*m.Accuracy, 100*m.Precision, 100*m.Recall, 100*m.F1, m.Confusion[0][0], m.Confusion[0][1], m.Confusion[1][0], m.Confusion[1][1])
}

func run() error {
	linearPath := flag.String("linear", "", "CSV линейной регрессии: area,price")
	logisticPath := flag.String("logistic", "", "CSV логистической регрессии: признаки,class")
	knnPath := flag.String("knn", "", "CSV KNN: признаки,class")
	epochs := flag.Int("epochs", 1000, "число итераций регрессий")
	rate := flag.Float64("rate", 0.01, "скорость обучения регрессий")
	k := flag.Int("k", 3, "число соседей KNN")
	query := flag.String("query", "2.5,3.1", "признаки нового объекта KNN через запятую")
	flag.Parse()
	data, err := loadData(*linearPath, linearCSV, "price")
	if err != nil {
		return err
	}
	train, test, err := split(data, 42)
	if err != nil {
		return err
	}
	m, history, err := TrainLinear(train, *rate, *epochs)
	if err != nil {
		return err
	}
	fmt.Printf("ЗАДАНИЕ 1. ЛИНЕЙНАЯ РЕГРЕССИЯ\nПрогноз цены квартиры по площади (учебные данные)\nОбъектов: %d; обучение: %d; тест: %d\nИтераций: %d; rate = %g; функция ошибки: MSE\nw = %.6f; bias = %.6f\nMSE обучения: %.6f -> %.6f\nПлощадь\tРеальная цена\tПредсказание\n", len(data), len(train), len(test), *epochs, *rate, m.Weight, m.Bias, history[0], history[len(history)-1])
	actual, predicted := make([]float64, len(test)), make([]float64, len(test))
	for i, row := range test {
		actual[i], predicted[i] = row.Y, m.Predict(row.X[0])
		fmt.Printf("%.1f\t%.2f\t%.2f\n", row.X[0], row.Y, predicted[i])
	}
	mae, mse, rmse := regressionMetrics(actual, predicted)
	fmt.Printf("MAE = %.6f\nMSE = %.6f\nRMSE = %.6f\n", mae, mse, rmse)
	data, err = loadData(*logisticPath, logisticCSV, "class")
	if err != nil {
		return err
	}
	train, test, err = split(data, 42)
	if err != nil {
		return err
	}
	logistic, loss, err := TrainLogistic(train, *rate, *epochs)
	if err != nil {
		return err
	}
	fmt.Printf("\nЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ\nКлассы: 0 (A), 1 (B); порог = 0.5\nОбъектов: %d; обучение: %d; тест: %d\nИтераций: %d; rate = %g; ошибка: binary cross-entropy\nВеса стандартизованных признаков: %v; bias = %.6f\nСредние обучения: %v; масштабы: %v\nОшибка обучения: %.6f -> %.6f\nОбъект\tФакт\tP(класс 1)\tКласс\n", len(data), len(train), len(test), *epochs, *rate, logistic.Weights, logistic.Bias, logistic.Scaler.Mean, logistic.Scaler.Scale, loss[0], loss[len(loss)-1])
	yTrue, yPred := make([]int, len(test)), make([]int, len(test))
	for i, row := range test {
		prob, class, err := logistic.Predict(row.X)
		if err != nil {
			return err
		}
		yTrue[i], yPred[i] = int(row.Y), class
		fmt.Printf("%v\t%d\t%.6f\t%d\n", row.X, int(row.Y), prob, class)
	}
	printMetrics(classificationMetrics(yTrue, yPred))
	data, err = loadData(*knnPath, knnCSV, "class")
	if err != nil {
		return err
	}
	train, test, err = split(data, 42)
	if err != nil {
		return err
	}
	knn, err := NewKNN(train, *k)
	if err != nil {
		return err
	}
	parts := strings.Split(*query, ",")
	x := make([]float64, len(parts))
	for j, part := range parts {
		value, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil || !finite(value) {
			return fmt.Errorf("query: ожидаются конечные числа через запятую")
		}
		x[j] = value
	}
	class, neighbors, err := knn.Predict(x)
	if err != nil {
		return err
	}
	fmt.Printf("\nЗАДАНИЕ 2. K NEAREST NEIGHBORS\nОбъектов: %d; признаков: %d; классов: 2\nОбучение: %d; тест: %d; K = %d\nМетрика: евклидово расстояние\nТестовый объект: %v\nБЛИЖАЙШИЕ СОСЕДИ\n№\tОбъект\tКласс\tРасстояние\n", len(data), len(data[0].X), len(train), len(test), *k, x)
	votes := [2]int{}
	for i, n := range neighbors {
		votes[int(n.Y)]++
		fmt.Printf("%d\t%v\t%d\t%.6f\n", i+1, n.X, int(n.Y), n.Distance)
	}
	fmt.Printf("Голосование: класс 0: %d; класс 1: %d\nПредсказанный класс: %d\nТЕСТИРОВАНИЕ НА ОТЛОЖЕННОЙ ВЫБОРКЕ\n", votes[0], votes[1], class)
	yTrue, yPred = make([]int, len(test)), make([]int, len(test))
	correct := 0
	for i, row := range test {
		prediction, _, err := knn.Predict(row.X)
		if err != nil {
			return err
		}
		yTrue[i], yPred[i] = int(row.Y), prediction
		if prediction == int(row.Y) {
			correct++
		}
		fmt.Printf("%v: факт %d, прогноз %d\n", row.X, int(row.Y), prediction)
	}
	fmt.Printf("Правильных: %d; ошибок: %d\n", correct, len(test)-correct)
	printMetrics(classificationMetrics(yTrue, yPred))
	allMetrics, err := leaveOneOut(data, *k)
	if err != nil {
		return err
	}
	allCorrect := allMetrics.Confusion[0][0] + allMetrics.Confusion[1][1]
	fmt.Printf("\nТЕСТИРОВАНИЕ ВСЕЙ ВЫБОРКИ (leave-one-out)\nКаждый объект исключается из обучения перед своим прогнозом\nПравильных: %d; ошибок: %d\n", allCorrect, len(data)-allCorrect)
	printMetrics(allMetrics)
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}
}
