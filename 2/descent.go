package main

import (
	"fmt"
	"math"
)

type Point struct{ X, Y float64 }

type Objective struct {
	Name    string
	Value   func(Point) float64
	Grad    func(Point) Point
	Minimum Point
}

type DescentStep struct {
	Point
	Value, GradNorm, Alpha float64
}

func dot(a, b Point) float64 { return a.X*b.X + a.Y*b.Y }

func move(p, g Point, alpha float64) Point {
	return Point{p.X - alpha*g.X, p.Y - alpha*g.Y}
}

// На выпуклом сечении минимум достигается при grad(f(p-alpha*g)) · g = 0.
func optimalStep(objective Objective, p, g Point) (float64, error) {
	directional := func(alpha float64) float64 { return dot(objective.Grad(move(p, g, alpha)), g) }
	left, right := 0.0, 1.0
	bracketed := false
	for i := 0; i < 60; i++ {
		d := directional(right)
		if !finite(d) {
			return 0, fmt.Errorf("поиск шага: неконечная направленная производная")
		}
		if d <= 0 {
			bracketed = true
			break
		}
		right *= 2
	}
	if !bracketed {
		return 0, fmt.Errorf("не удалось ограничить минимум вдоль антиградиента")
	}
	for i := 0; i < 60; i++ {
		mid := (left + right) / 2
		d := directional(mid)
		if !finite(d) {
			return 0, fmt.Errorf("поиск шага: неконечная направленная производная")
		}
		if d > 0 {
			left = mid
		} else {
			right = mid
		}
	}
	return (left + right) / 2, nil
}

func Descent(objective Objective, start Point, rate, eps float64, limit int, steepest bool) ([]DescentStep, error) {
	if objective.Value == nil || objective.Grad == nil || !finite(start.X) || !finite(start.Y) ||
		!finite(rate) || rate <= 0 || !finite(eps) || eps <= 0 || limit <= 0 {
		return nil, fmt.Errorf("некорректные параметры спуска")
	}
	p, alpha := start, 0.0
	steps := make([]DescentStep, 0)
	for i := 0; i <= limit; i++ {
		g := objective.Grad(p)
		value, norm := objective.Value(p), math.Hypot(g.X, g.Y)
		if !finite(p.X) || !finite(p.Y) || !finite(value) || !finite(norm) {
			return steps, fmt.Errorf("спуск расходится: неконечное значение")
		}
		steps = append(steps, DescentStep{p, value, norm, alpha})
		if norm <= eps {
			return steps, nil
		}
		if i == limit {
			break
		}
		alpha = rate
		if steepest {
			var err error
			alpha, err = optimalStep(objective, p, g)
			if err != nil {
				return steps, err
			}
		}
		next := move(p, g, alpha)
		if next == p {
			return steps, fmt.Errorf("достигнут предел точности координат")
		}
		p = next
	}
	return steps, fmt.Errorf("спуск: превышен лимит %d итераций", limit)
}

var bowl = Objective{
	Name:    "(x - 2)^2 + 2(y + 1)^2 + 3",
	Value:   func(p Point) float64 { return (p.X-2)*(p.X-2) + 2*(p.Y+1)*(p.Y+1) + 3 },
	Grad:    func(p Point) Point { return Point{2 * (p.X - 2), 4 * (p.Y + 1)} },
	Minimum: Point{2, -1},
}

var comparisonBowl = Objective{
	Name:    "4x^2 + y^2 - 4x - 2y + 5",
	Value:   func(p Point) float64 { return 4*p.X*p.X + p.Y*p.Y - 4*p.X - 2*p.Y + 5 },
	Grad:    func(p Point) Point { return Point{8*p.X - 4, 2*p.Y - 2} },
	Minimum: Point{0.5, 1},
}
