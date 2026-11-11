package main

import (
	"fmt"
	"math"
)

type RootStep struct {
	Iteration                            int
	A, B, X, FX, Derivative, Next, Delta float64
}

type RootResult struct {
	X          float64
	Iterations int
	Steps      []RootStep
}

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

func rootParams(f func(float64) float64, eps float64, limit int, points ...float64) error {
	if f == nil || !finite(eps) || eps <= 0 || limit <= 0 {
		return fmt.Errorf("нужны функция, epsilon > 0 и положительный лимит итераций")
	}
	for _, x := range points {
		if !finite(x) || !finite(f(x)) {
			return fmt.Errorf("неконечная точка или значение функции")
		}
	}
	return nil
}

func Bisection(f func(float64) float64, a, b, eps float64, limit int) (RootResult, error) {
	r := RootResult{}
	if err := rootParams(f, eps, limit, a, b); err != nil {
		return r, err
	}
	fa, fb := f(a), f(b)
	if a >= b {
		return r, fmt.Errorf("требуется a < b")
	}
	if math.Abs(fa) <= eps {
		r.X = a
		return r, nil
	}
	if math.Abs(fb) <= eps {
		r.X = b
		return r, nil
	}
	if math.Signbit(fa) == math.Signbit(fb) {
		return r, fmt.Errorf("на концах интервала нет смены знака")
	}
	for i := 1; i <= limit; i++ {
		x := a/2 + b/2
		fx := f(x)
		if !finite(fx) {
			return r, fmt.Errorf("дихотомия: функция не определена внутри интервала")
		}
		r.X, r.Iterations = x, i
		r.Steps = append(r.Steps, RootStep{Iteration: i, A: a, B: b, X: x, FX: fx})
		if math.Abs(fx) <= eps {
			return r, nil
		}
		if x == a || x == b {
			return r, fmt.Errorf("дихотомия: достигнут предел точности float64")
		}
		if math.Signbit(fa) == math.Signbit(fx) {
			a, fa = x, fx
		} else {
			b = x
		}
	}
	return r, fmt.Errorf("дихотомия: превышен лимит итераций")
}

func Secant(f func(float64) float64, a, b, eps float64, limit int) (RootResult, error) {
	r := RootResult{}
	if err := rootParams(f, eps, limit, a, b); err != nil {
		return r, err
	}
	fa, fb := f(a), f(b)
	if math.Abs(fa) <= eps {
		r.X = a
		return r, nil
	}
	if math.Abs(fb) <= eps {
		r.X = b
		return r, nil
	}
	for i := 1; i <= limit; i++ {
		if fb == fa {
			return r, fmt.Errorf("секущие: нулевой знаменатель")
		}
		x := b - fb*(b-a)/(fb-fa)
		fx := f(x)
		if !finite(x) || !finite(fx) {
			return r, fmt.Errorf("секущие: неконечное значение")
		}
		r.X, r.Iterations = x, i
		r.Steps = append(r.Steps, RootStep{Iteration: i, X: x, FX: fx})
		if math.Abs(fx) <= eps {
			return r, nil
		}
		a, fa, b, fb = b, fb, x, fx
	}
	return r, fmt.Errorf("секущие: превышен лимит итераций")
}

func Newton(f, derivative func(float64) float64, x, eps float64, limit int) (RootResult, error) {
	r := RootResult{X: x}
	if err := rootParams(f, eps, limit, x); err != nil {
		return r, err
	}
	if derivative == nil {
		return r, fmt.Errorf("не задана производная")
	}
	for i := 0; i <= limit; i++ {
		fx, d := f(x), derivative(x)
		if !finite(x) || !finite(fx) || !finite(d) {
			return r, fmt.Errorf("Ньютон: неконечное значение")
		}
		r.X, r.Iterations = x, i
		r.Steps = append(r.Steps, RootStep{Iteration: i, X: x, FX: fx, Derivative: d})
		if math.Abs(fx) <= eps {
			return r, nil
		}
		if d == 0 {
			return r, fmt.Errorf("Ньютон: нулевая производная")
		}
		x -= fx / d
	}
	return r, fmt.Errorf("Ньютон: превышен лимит итераций")
}

func FixedPoint(f, phi func(float64) float64, x, eps float64, limit int) (RootResult, error) {
	r := RootResult{X: x}
	if err := rootParams(f, eps, limit, x); err != nil {
		return r, err
	}
	if phi == nil {
		return r, fmt.Errorf("не задана phi")
	}
	for i := 0; i <= limit; i++ {
		next, fx := phi(x), f(x)
		if !finite(next) || !finite(fx) {
			return r, fmt.Errorf("простая итерация: неконечное значение")
		}
		r.X, r.Iterations = x, i
		r.Steps = append(r.Steps, RootStep{Iteration: i, X: x, FX: fx, Next: next, Delta: math.Abs(next - x)})
		if math.Abs(fx) <= eps {
			return r, nil
		}
		x = next
	}
	return r, fmt.Errorf("простая итерация: превышен лимит итераций")
}
