package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"
)

func printDescent(name string, objective Objective, steps []DescentStep, elapsed time.Duration) {
	fmt.Printf("\n%s\nИтерация\tx\ty\tf(x,y)\t|grad|\talpha\n", name)
	for i, s := range steps {
		fmt.Printf("%d\t%.9f\t%.9f\t%.9f\t%.9g\t%.9g\n", i, s.X, s.Y, s.Value, s.GradNorm, s.Alpha)
	}
	last := steps[len(steps)-1]
	fmt.Printf("МИНИМУМ НАЙДЕН: (%.9f, %.9f); f = %.9f\nИтераций: %d; норма градиента = %.9g\n", last.X, last.Y, last.Value, len(steps)-1, last.GradNorm)
	fmt.Printf("Ошибки координат: |x-x*| = %.9g; |y-y*| = %.9g\nВремя: %.3f ms\n", math.Abs(last.X-objective.Minimum.X), math.Abs(last.Y-objective.Minimum.Y), float64(elapsed)/float64(time.Millisecond))
}

func formatIterationRatio(gradientSteps, steepestSteps int) string {
	if steepestSteps == 0 {
		return "неприменимо"
	}
	return fmt.Sprintf("%.3f", float64(gradientSteps)/float64(steepestSteps))
}

func run() error {
	eps := flag.Float64("epsilon", 1e-6, "точность по невязке корня / норме градиента")
	rootLimit := flag.Int("root-limit", 100, "максимум итераций поиска корня")
	descentLimit := flag.Int("descent-limit", 1000, "максимум итераций спуска")
	rate := flag.Float64("rate", 0.1, "постоянный шаг градиентного спуска")
	out := flag.String("out", "results", "папка графиков и CSV")
	flag.Parse()
	f := func(x float64) float64 { return x*x*x - x - 2 }
	df := func(x float64) float64 { return 3*x*x - 1 }
	phi := func(x float64) float64 { return math.Cbrt(x + 2) }
	fmt.Printf("ЗАДАНИЕ 1. РЕШЕНИЕ НЕЛИНЕЙНОГО УРАВНЕНИЯ\nf(x) = x^3 - x - 2 = 0\n[a,b] = [1,2]; f(a) = %g; f(b) = %g\nepsilon = %g; лимит = %d\nПроизводная: 3x^2 - 1; x0 Ньютона = 2\nphi(x) = cbrt(x+2); x0 простой итерации = 1.5\n", f(1), f(2), *eps, *rootLimit)
	names := []string{"ДИХОТОМИЯ", "ХОРДЫ (СЕКУЩИЕ)", "НЬЮТОН", "ПРОСТАЯ ИТЕРАЦИЯ"}
	solvers := []func() (RootResult, error){
		func() (RootResult, error) { return Bisection(f, 1, 2, *eps, *rootLimit) },
		func() (RootResult, error) { return Secant(f, 1, 2, *eps, *rootLimit) },
		func() (RootResult, error) { return Newton(f, df, 2, *eps, *rootLimit) },
		func() (RootResult, error) { return FixedPoint(f, phi, 1.5, *eps, *rootLimit) },
	}
	results := make([]RootResult, len(solvers))
	for j, solve := range solvers {
		r, err := solve()
		if err != nil {
			return err
		}
		results[j] = r
		fmt.Printf("\n%s\n", names[j])
		switch j {
		case 0:
			fmt.Println("Итерация\ta\tb\tc\tf(c)")
		case 1:
			fmt.Println("Итерация\tx\tf(x)")
		case 2:
			fmt.Println("Итерация\tx\tf(x)\tf'(x)")
		case 3:
			fmt.Println("Итерация\tx\tphi(x)\t|x(n+1)-x(n)|")
		}
		for _, s := range r.Steps {
			switch j {
			case 0:
				fmt.Printf("%d\t%.9f\t%.9f\t%.9f\t%.9g\n", s.Iteration, s.A, s.B, s.X, s.FX)
			case 1:
				fmt.Printf("%d\t%.9f\t%.9g\n", s.Iteration, s.X, s.FX)
			case 2:
				fmt.Printf("%d\t%.9f\t%.9g\t%.9g\n", s.Iteration, s.X, s.FX, s.Derivative)
			case 3:
				fmt.Printf("%d\t%.9f\t%.9f\t%.9g\n", s.Iteration, s.X, s.Next, s.Delta)
			}
		}
		fmt.Printf("Корень: %.10f; f(x) = %.9g; итераций: %d\n", r.X, f(r.X), r.Iterations)
	}
	fmt.Println("\nСРАВНЕНИЕ МЕТОДОВ\nМетод\tКорень\tИтерации\tНевязка |f(x)|")
	for i, r := range results {
		fmt.Printf("%s\t%.10f\t%d\t%.9g\n", names[i], r.X, r.Iterations, math.Abs(f(r.X)))
	}
	if err := os.MkdirAll(*out, 0755); err != nil {
		return err
	}
	fmt.Printf("\nЗАДАНИЕ 2. ГРАДИЕНТНЫЙ СПУСК\nf(x,y) = %s\nГрадиент = (2(x-2), 4(y+1))\nНачало: (3,4); alpha = %g; epsilon = %g; лимит = %d\nТеоретический минимум: (2,-1); f = %g\n", bowl.Name, *rate, *eps, *descentLimit, bowl.Value(bowl.Minimum))
	started := time.Now()
	path, err := Descent(bowl, Point{3, 4}, *rate, *eps, *descentLimit, false)
	elapsed := time.Since(started)
	if err != nil {
		return err
	}
	printDescent("ГРАДИЕНТНЫЙ СПУСК", bowl, path, elapsed)
	if err := saveTrajectory(filepath.Join(*out, "gradient.csv"), path); err != nil {
		return err
	}
	if err := savePlot(filepath.Join(*out, "gradient.png"), bowl, [][]DescentStep{path}, []string{"Gradient descent"}); err != nil {
		return err
	}
	fmt.Printf("\nЗАДАНИЕ 3. СРАВНЕНИЕ МЕТОДОВ СПУСКА\nf(x,y) = %s\nГрадиент = (8x-4, 2y-2)\nНачало: (5,5); alpha обычного спуска = %g; epsilon = %g; лимит = %d\nТеоретический минимум: (0.5,1); f = %g\n", comparisonBowl.Name, *rate, *eps, *descentLimit, comparisonBowl.Value(comparisonBowl.Minimum))
	paths := make([][]DescentStep, 2)
	times := make([]time.Duration, 2)
	descentNames := []string{"ГРАДИЕНТНЫЙ СПУСК", "НАИСКОРЕЙШИЙ СПУСК"}
	for i := range paths {
		started := time.Now()
		paths[i], err = Descent(comparisonBowl, Point{5, 5}, *rate, *eps, *descentLimit, i == 1)
		times[i] = time.Since(started)
		if err != nil {
			return err
		}
		printDescent(descentNames[i], comparisonBowl, paths[i], times[i])
		if err := saveTrajectory(filepath.Join(*out, []string{"comparison_gradient.csv", "comparison_steepest.csv"}[i]), paths[i]); err != nil {
			return err
		}
	}
	if err := savePlot(filepath.Join(*out, "comparison.png"), comparisonBowl, paths, []string{"Gradient descent", "Steepest descent"}); err != nil {
		return err
	}
	fmt.Println("\nИТОГОВОЕ СРАВНЕНИЕ\nМетод\tИтерации\tf(x*)\tОшибка координат (L2)\tВремя, ms")
	for i, steps := range paths {
		s := steps[len(steps)-1]
		fmt.Printf("%s\t%d\t%.9f\t%.9g\t%.3f\n", descentNames[i], len(steps)-1, s.Value, math.Hypot(s.X-0.5, s.Y-1), float64(times[i])/float64(time.Millisecond))
	}
	fmt.Printf("Разница итераций: %d; ускорение по числу итераций: %s\nГрафики и траектории: %s\n", len(paths[0])-len(paths[1]), formatIterationRatio(len(paths[0])-1, len(paths[1])-1), *out)
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}
}
