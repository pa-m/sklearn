package preprocessing

func intrange(n int) []int {
	r := make([]int, n)
	for i := range r {
		r[i] = i
	}
	return r
}

func yieldPoolIndices(ch chan<- []int, pool, indices []int) {
	a := make([]int, len(indices))
	for ii, i := range indices {
		a[ii] = pool[i]
	}
	ch <- a
}
func combinations(pool []int, r int) chan []int {
	n := len(pool)
	ch := make(chan []int)
	indices := intrange(r)

	// yield tuple(pool[i] for i in indices)
	go func() {
		if r > n {
			close(ch)
			return
		}
		yieldPoolIndices(ch, pool, indices)
		for {
			var i int
			loops := 0
			//for i in reversed(range(r)):
			for i = r - 1; i >= 0; i-- {
				loops++
				if indices[i] != i+n-r {
					break
				}
			}
			if i < 0 {
				break
			}
			indices[i]++
			for j := i + 1; j < r; j++ {
				indices[j] = indices[j-1] + 1
			}
			yieldPoolIndices(ch, pool, indices)
		}
		close(ch)
	}()
	return ch
}

func combinationsWithReplacement(pool []int, r int) chan []int {
	n := len(pool)
	ch := make(chan []int)
	indices := make([]int, r)
	go func() {
		if n == 0 && r > 0 {
			close(ch)
			return
		}
		yieldPoolIndices(ch, pool, indices)
		for {
			var i int
			for i = r - 1; i >= 0; i-- {
				if indices[i] != n-1 {
					break
				}
			}
			if i < 0 {
				close(ch)
				return
			}
			indi := indices[i] + 1
			for ii := i; ii < r; ii++ {
				indices[ii] = indi
			}
			yieldPoolIndices(ch, pool, indices)
		}
	}()
	return ch
}

func bincount(a []int, minlength int) []int {
	ret := make([]int, minlength)
	for _, ai := range a {
		if ai >= 0 && ai < minlength {
			ret[ai]++
		}
	}
	return ret
}
