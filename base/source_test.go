package base

import (
	"golang.org/x/exp/rand"

	"fmt"
	"github.com/pa-m/randomkit"
	"testing"
)

var (
	_ = []rand.Source{&randomkit.RKState{}, &LockedSource{}}
	_ = []Float64er{&randomkit.RKState{}}
	_ = []NormFloat64er{&randomkit.RKState{}}
	_ = []SourceCloner{&LockedSource{}}
)

func TestSource(t *testing.T) {
	s := NewSource(7)
	s2 := s.SourceClone()
	var a [5]float64
	for i := range a {
		a[i] = s.Float64()
	}
	expected := "[0.07630829 0.77991879 0.43840923 0.72346518 0.97798951]"
	actual := fmt.Sprintf("%.8f", a)
	if expected != actual {
		t.Errorf("expected:\n%s\ngot:\n%s", expected, actual)
	}
	for i := range a {
		a[i] = s2.(Float64er).Float64()
	}
	actual = fmt.Sprintf("%.8f", a)
	if expected != actual {
		t.Errorf("expected:\n%s\ngot:\n%s", expected, actual)
	}
}
