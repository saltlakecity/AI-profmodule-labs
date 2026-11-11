package main

import (
	"math"
	"testing"
)

func TestIterationRatio(t *testing.T) {
	for _, test := range []struct {
		gradient, steepest int
		want               string
	}{{72, 15, "4.800"}, {0, 0, "неприменимо"}, {5, 0, "неприменимо"}, {0, 5, "0.000"}} {
		if got := formatIterationRatio(test.gradient, test.steepest); got != test.want {
			t.Fatalf("отношение (%d, %d) = %q, ожидается %q", test.gradient, test.steepest, got, test.want)
		}
	}
}

func TestRootMethods(t *testing.T) {
	f := func(x float64) float64 { return x*x*x - x - 2 }
	df := func(x float64) float64 { return 3*x*x - 1 }
	phi := func(x float64) float64 { return math.Cbrt(x + 2) }
	methods := []struct {
		name  string
		solve func() (RootResult, error)
	}{
		{"bisection", func() (RootResult, error) { return Bisection(f, 1, 2, 1e-9, 100) }},
		{"secant", func() (RootResult, error) { return Secant(f, 1, 2, 1e-9, 100) }},
		{"newton", func() (RootResult, error) { return Newton(f, df, 2, 1e-9, 100) }},
		{"fixed", func() (RootResult, error) { return FixedPoint(f, phi, 1.5, 1e-9, 100) }},
	}
	for _, method := range methods {
		t.Run(method.name, func(t *testing.T) {
			r, err := method.solve()
			if err != nil || math.Abs(f(r.X)) > 1e-9 || math.Abs(r.X-1.5213797068045676) > 1e-8 {
				t.Fatalf("%+v, %v", r, err)
			}
		})
	}
	// Второе уравнение проверяет, что алгоритмы не привязаны к примеру.
	q := func(x float64) float64 { return x*x - 2 }
	r, err := Newton(q, func(x float64) float64 { return 2 * x }, 1, 1e-10, 100)
	if err != nil || math.Abs(r.X-math.Sqrt2) > 1e-9 {
		t.Fatalf("sqrt(2): %v, %v", r, err)
	}
}

func TestRootFailuresAndEndpoint(t *testing.T) {
	f := func(x float64) float64 { return x*x + 1 }
	if _, err := Bisection(f, -1, 1, 1e-6, 10); err == nil {
		t.Fatal("нет смены знака")
	}
	if _, err := Secant(f, -1, 1, 1e-6, 10); err == nil {
		t.Fatal("нулевой знаменатель")
	}
	if _, err := Newton(f, func(float64) float64 { return 0 }, 0, 1e-6, 10); err == nil {
		t.Fatal("нулевая производная")
	}
	if _, err := FixedPoint(f, func(x float64) float64 { return x + 1 }, 0, 1e-6, 3); err == nil {
		t.Fatal("нет сходимости")
	}
	r, err := Bisection(func(x float64) float64 { return x - 1 }, 1, 2, 1e-6, 10)
	if err != nil || r.X != 1 || r.Iterations != 0 {
		t.Fatalf("корень на границе: %+v %v", r, err)
	}
	if _, err := Bisection(f, 0, 1, math.NaN(), 10); err == nil {
		t.Fatal("принят NaN")
	}
	if _, err := Newton(func(float64) float64 { return math.NaN() }, func(float64) float64 { return 1 }, 0, 1e-6, 10); err == nil {
		t.Fatal("принята неконечная функция")
	}
}

func TestDescentAndOptimalStep(t *testing.T) {
	for _, objective := range []Objective{bowl, comparisonBowl} {
		for _, steepest := range []bool{false, true} {
			steps, err := Descent(objective, Point{5, 5}, 0.1, 1e-6, 1000, steepest)
			if err != nil {
				t.Fatal(err)
			}
			last := steps[len(steps)-1]
			if last.GradNorm > 1e-6 || math.Hypot(last.X-objective.Minimum.X, last.Y-objective.Minimum.Y) > 1e-6 {
				t.Fatalf("не найден минимум: %+v", last)
			}
			for i := 1; i < len(steps); i++ {
				if steps[i].Value > steps[i-1].Value+1e-12 {
					t.Fatal("функция возрастает")
				}
				if steepest {
					prevG := objective.Grad(steps[i-1].Point)
					g := objective.Grad(steps[i].Point)
					if math.Abs(dot(prevG, g)) > 1e-8*max(1, dot(prevG, prevG)) {
						t.Fatal("шаг не минимизирует сечение")
					}
				}
			}
		}
	}
	if comparisonBowl.Value(comparisonBowl.Minimum) != 3 {
		t.Fatal("минимум примера равен 3")
	}
}

func TestDescentFailuresAndAlreadyAtMinimum(t *testing.T) {
	s, err := Descent(bowl, bowl.Minimum, 0.1, 1e-6, 100, false)
	if err != nil || len(s) != 1 {
		t.Fatalf("%v %v", s, err)
	}
	if _, err := Descent(bowl, Point{3, 4}, 10, 1e-6, 20, false); err == nil {
		t.Fatal("расходящийся спуск должен завершиться ошибкой")
	}
	if _, err := Descent(bowl, Point{3, 4}, 0.1, -1, 20, false); err == nil {
		t.Fatal("отрицательная точность")
	}
	if _, err := Descent(bowl, Point{math.Inf(1), 0}, 0.1, 1e-6, 20, false); err == nil {
		t.Fatal("неконечная точка")
	}
}
