package preprocessing

import (
	"fmt"
	"testing"
)

func Test_combinations(t *testing.T) {
	fixtures := []struct {
		pool     []int
		r        int
		expected string
	}{
		{[]int{0, 1, 2}, 2, "[[0 1] [0 2] [1 2]]"},
		{[]int{0, 1, 2, 3}, 3, "[[0 1 2] [0 1 3] [0 2 3] [1 2 3]]"},
	}
	chan2slice := func(ch chan []int) [][]int {
		a := [][]int{}
		for c := range ch {
			a = append(a, c)
		}
		return a
	}
	for _, f := range fixtures {
		actual := fmt.Sprint(chan2slice(combinations(f.pool, f.r)))
		if actual != f.expected {
			t.Errorf("expected %s, got %s", f.expected, actual)
		}
	}
}
func Test_combinations_wr(t *testing.T) {
	fixtures := []struct {
		pool     []int
		r        int
		expected string
	}{
		{[]int{0, 1, 2, 3}, 2, "[[0 0] [0 1] [0 2] [0 3] [1 1] [1 2] [1 3] [2 2] [2 3] [3 3]]"},
	}
	chan2slice := func(ch chan []int) [][]int {
		a := [][]int{}
		for c := range ch {
			a = append(a, c)
		}
		return a
	}
	for _, f := range fixtures {
		actual := fmt.Sprint(chan2slice(combinationsWithReplacement(f.pool, f.r)))
		if actual != f.expected {
			t.Errorf("expected %s, got %s", f.expected, actual)
		}
	}
}
