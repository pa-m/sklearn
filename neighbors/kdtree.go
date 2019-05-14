package neighbors

import (
	"fmt"
	"math"
	"runtime"
	"sort"

	"github.com/pa-m/sklearn/base"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Rectangle Hyperrectangle class.    Represents a Cartesian product of intervals.
type Rectangle struct{ Maxes, Mins []float64 }

// NewRectangle ...
func NewRectangle(Maxes, Mins []float64) *Rectangle {
	return &Rectangle{Maxes: Maxes, Mins: Mins}
}

// String ...
func (r *Rectangle) String() string {
	s := "<Rectangle "
	for axis := 0; axis < len(r.Maxes); axis++ {
		if axis > 0 {
			s += ", "
		}
		s += fmt.Sprint(r.Mins[axis], r.Maxes[axis])
	}
	s += ">"
	return s
}

// Volume ...
func (r *Rectangle) Volume() float64 {
	v := 1.
	for axis := 0; axis < len(r.Maxes); axis++ {
		v *= r.Maxes[axis] - r.Mins[axis]
	}
	return v
}

// Split Produce two hyperrectangles by splitting.
// In general, if you need to compute maximum and minimum
// distances to the children, it can be done more efficiently
// by updating the maximum and minimum distances to the parent.
// Parameters
// ----------
// d : int
// 	Axis to split hyperrectangle along.
// split : float
// 	Position along axis `d` to split at.
func (r *Rectangle) Split(d int, split float64) (less, greater *Rectangle) {
	cp := copyFloatSlice
	mid := cp(r.Maxes)
	mid[d] = split
	less = NewRectangle(r.Mins, mid)
	mid = cp(r.Mins)
	mid[d] = split
	greater = NewRectangle(mid, r.Maxes)
	return less, greater
}

// MinDistancePoint Return the minimum distance between input and points in the hyperrectangle.
// Parameters
// x : array_like  	Input.
// p : float, optional 	Input.
func (r *Rectangle) MinDistancePoint(x []float64, p float64) float64 {
	l := len(x)
	v := make([]float64, l)
	for d := 0; d < l; d++ {
		v[d] = math.Max(0, math.Max(r.Mins[d]-x[d], x[d]-r.Maxes[d]))
	}
	return mat.Norm(mat.NewVecDense(l, v), p)
}

// MaxDistancePoint return the maximum distance between input and points in the hyperrectangle.
// Parameters
// x : array_like 	Input array.
// p : float, optional	Input.
func (r *Rectangle) MaxDistancePoint(x []float64, p float64) float64 {
	l := len(x)
	v := make([]float64, l)
	for d := 0; d < l; d++ {
		v[d] = math.Max(r.Maxes[d]-x[d], x[d]-r.Mins[d])
	}
	return mat.Norm(mat.NewVecDense(l, v), p)
}

// MinDistanceRectangle compute the minimum distance between points in the two hyperrectangles.
// Parameters
// other : hyperrectangle Input.
// p : float Input.
func (r *Rectangle) MinDistanceRectangle(other *Rectangle, p float64) float64 {
	l := len(other.Maxes)
	v := make([]float64, l)
	for d := 0; d < l; d++ {
		v[d] = math.Max(0, math.Max(r.Mins[d]-other.Maxes[d], other.Mins[d]-r.Maxes[d]))
	}
	return mat.Norm(mat.NewVecDense(l, v), p)
}

// MaxDistanceRectangle compute the maximum distance between points in the two hyperrectangles.
// Parameters
// other : hyperrectangle     Input.
// p : float, optional      Input.
func (r *Rectangle) MaxDistanceRectangle(other *Rectangle, p float64) float64 {
	l := len(other.Maxes)
	v := make([]float64, l)
	for d := 0; d < l; d++ {
		v[d] = math.Max(r.Maxes[d]-other.Mins[d], other.Maxes[d]-r.Mins[d])
	}
	return mat.Norm(mat.NewVecDense(l, v), p)
}

// KDTree for quick nearest-neighbor lookup
// This class provides an index into a set of k-dimensional points which
// can be used to rapidly look up the nearest neighbors of any point.
// Parameters
// Data : (N,K) mat.Dense The data points to be indexed. This array is not copied, and
// 	so modifying this data will result in bogus results.
// Leafsize : int, optional The number of points at which the algorithm switches over to
// 	brute-force.  Has to be positive.
// Notes
// The algorithm used is described in Maneewongvatana and Mount 1999.
// The general idea is that the kd-tree is a binary tree, each of whose
// nodes represents an axis-aligned hyperrectangle. Each node specifies
// an axis and splits the set of points based on whether their coordinate
// along that axis is greater than or less than a particular value.
// During construction, the axis and splitting point are chosen by the
// "sliding midpoint" rule, which ensures that the cells do not all
// become long and thin.
// The tree can be queried for the r closest neighbors of any given point
// (optionally returning only those within some maximum distance of the
// point). It can also be queried, with a substantial gain in efficiency,
// for the r approximate closest neighbors.
// For large dimensions (20 is already large) do not expect this to run
// significantly faster than brute force. High-dimensional nearest-neighbor
// queries are a substantial open problem in computer science.
// The tree also supports all-neighbors queries, both with arrays of points
// and with other kd-trees. These do use a reasonably efficient algorithm,
// but the kd-tree is not necessarily the best data structure for this
// sort of calculation.
type KDTree struct {
	Data        *mat.Dense
	LeafSize    int
	Maxes, Mins []float64
	Tree        Node
}

// NewKDTree ...
func NewKDTree(data mat.Matrix, LeafSize int) *KDTree {
	tr := &KDTree{
		Data:     mat.DenseCopyOf(data),
		LeafSize: LeafSize,
	}
	if tr.LeafSize < 1 {
		tr.LeafSize = 1
	}
	n, m := data.Dims()
	tr.Maxes = make([]float64, m)
	tr.Mins = make([]float64, m)
	copy(tr.Maxes, tr.Data.RawRowView(0))
	copy(tr.Mins, tr.Data.RawRowView(0))
	rawdata := tr.Data.RawMatrix()
	base.Parallelize(runtime.NumCPU(), m, func(th, start, end int) {
		for j := start; j < end; j++ {
			for i, rowpos := 1, rawdata.Stride; i < n; i, rowpos = i+1, rowpos+rawdata.Stride {
				v := rawdata.Data[rowpos+j]
				if v > tr.Maxes[j] {
					tr.Maxes[j] = v
				}
				if v < tr.Mins[j] {
					tr.Mins[j] = v
				}
			}
		}
	})

	tr.Tree = tr._build(_arange(n), tr.Maxes, tr.Mins)
	return tr
}

func _arange(n int) (a []int) {
	a = make([]int, n)
	for i := range a {
		a[i] = i
	}
	return
}

// Node ...
type Node interface {
	IsLeaf() bool
}

// LeafNode ...
type LeafNode struct {
	Node
	idx      []int
	children int
}

// IsLeaf ...
func (*LeafNode) IsLeaf() bool { return true }

// InnerNode ...
type InnerNode struct {
	Node
	splitDim      int
	split         float64
	less, greater Node
}

// IsLeaf ...
func (*InnerNode) IsLeaf() bool { return false }

func copyFloatSlice(a []float64) []float64 {
	out := make([]float64, len(a), cap(a))
	copy(out, a)
	return out
}

func (tr *KDTree) _build(idx []int, maxes, mins []float64) Node {
	if len(idx) <= tr.LeafSize {
		return &LeafNode{idx: idx}
	}
	var d int
	M := math.Inf(-1)
	for i := 0; i < len(maxes); i++ {
		diff := maxes[i] - mins[i]
		if diff > M {
			d = i
			M = diff
		}
	}
	maxval, minval := maxes[d], mins[d]
	if maxval == minval {
		// # all points are identical; warn user?
		return &LeafNode{idx: idx}
	}
	// data=tr.Data[idx,d]
	// # sliding midpoint rule; see Maneewongvatana and Mount 1999
	// # for arguments that this is a good idea.
	split := (maxval + minval) / 2.
	lgfill := func(split float64) ([]int, []int) {
		lessIdx, greaterIdx := make([]int, 0), make([]int, 0)
		for _, idx1 := range idx {
			if tr.Data.At(idx1, d) <= split {
				lessIdx = append(lessIdx, idx1)
			}
			if tr.Data.At(idx1, d) > split {
				greaterIdx = append(greaterIdx, idx1)
			}
		}
		return lessIdx, greaterIdx
	}
	lessIdx, greaterIdx := lgfill(split)
	if len(lessIdx) == 0 {
		M := math.Inf(1)
		for _, i := range idx {
			if v := tr.Data.At(i, d); v < M {
				M = v
			}
		}
		split = M
		lessIdx, greaterIdx = lgfill(split)
	}
	if len(greaterIdx) == 0 {
		M := math.Inf(-1)
		for _, i := range idx {
			if v := tr.Data.At(i, d); v > M {
				M = v
			}
		}
		split = M
		lessIdx, greaterIdx = lgfill(split)
	}
	if len(lessIdx) == 0 {
		// # _still_ zero? all must have the same value
		split = tr.Data.At(idx[0], d)
		lessIdx = _arange(len(idx) - 1)
		greaterIdx = []int{len(idx) - 1}
	}
	lessmaxes := copyFloatSlice(maxes)
	lessmaxes[d] = split
	greatermins := copyFloatSlice(mins)
	greatermins[d] = split
	return &InnerNode{splitDim: d, split: split, less: tr._build(lessIdx, lessmaxes, mins), greater: tr._build(greaterIdx, maxes, greatermins)}
}

func (tr *KDTree) _query(X mat.Vector, k int, eps, p float64, distanceUpperBound float64, callback func(ith int, dist float64, indice int)) {
	cx := X.Len()
	sideDistances := make([]float64, cx)
	for j := 0; j < cx; j++ {
		x := X.AtVec(j)
		d := math.Max(0, math.Max(x-tr.Maxes[j], tr.Mins[j]-x))
		if !math.IsInf(p, 1) {
			d = math.Pow(d, p)
		}
		sideDistances[j] = d
	}

	var minDistance float64
	if !math.IsInf(p, 1) {
		minDistance = floats.Sum(sideDistances)
	} else {
		minDistance = floats.Max(sideDistances)
	}
	// # priority queue for chasing nodes
	// # entries are:
	// #  minimum distance between the cell and the target
	// #  distances between the nearest side of the cell and the target
	// #  the head node of the cell
	type qEle struct {
		minDistance   float64
		sideDistances []float64
		headNode      Node
	}
	// type qSlice []qEle

	q := []qEle{{minDistance, sideDistances, tr.Tree}}
	qLess := func(i, j int) bool { return q[i].minDistance < q[j].minDistance }
	qHeappush := func(e qEle) {
		q = append(q, e)
		if len(q) >= 2 {
			sort.Slice(q, qLess)
		}
	}
	qHeappop := func() qEle { e := q[0]; q = q[1:]; return e }

	// # priority queue for the nearest neighbors
	// # furthest known neighbor first
	// # entries are (-distance**p, i)
	type nEle struct {
		float64
		int
	}
	neighbors := []nEle{}
	nLess := func(i, j int) bool { return neighbors[i].float64 < neighbors[j].float64 }
	nHeappush := func(e nEle) {
		neighbors = append(neighbors, e)
		if len(neighbors) >= 2 {
			sort.Slice(neighbors, nLess)
		}
	}
	nHeappop := func() nEle { e := neighbors[0]; neighbors = neighbors[1:]; return e }

	var epsfac float64
	if eps == 0 {
		epsfac = 1.
	} else if math.IsInf(p, 1) {
		epsfac = 1 / (1 + eps)
	} else {
		epsfac = 1. / math.Pow(1+eps, p)
	}
	if !math.IsInf(p, 1) && !math.IsInf(distanceUpperBound, 1) && p != 1. {
		distanceUpperBound = math.Pow(distanceUpperBound, p)
	}
	for len(q) > 0 {
		qE := qHeappop()
		minDistance, sideDistances, node := qE.minDistance, qE.sideDistances, qE.headNode
		if node.IsLeaf() {
			// brute-force
			idx := node.(*LeafNode).idx

			for _, fitSample := range idx {
				ds := MinkowskiDistanceP(X, tr.Data.RowView(fitSample), p)
				if ds < distanceUpperBound {
					if len(neighbors) == k {
						nHeappop()
					}
				}
				nHeappush(nEle{float64: -ds, int: fitSample})
				if len(neighbors) == k {
					distanceUpperBound = -neighbors[0].float64
				}

			}

		} else {
			// # we don't push cells that are too far onto the queue at all,
			// # but since the distance_upper_bound decreases, we might get
			// # here even if the cell's too far
			if minDistance > distanceUpperBound*epsfac {
				// # since this is the nearest cell, we're done, bail out
				break
			}
			// # compute minimum distances to the children and push them on
			innernode := node.(*InnerNode)
			x := X.AtVec(innernode.splitDim)
			var near, far Node
			if x < innernode.split {
				near, far = innernode.less, innernode.greater
			} else {
				near, far = innernode.greater, innernode.less
			}
			//# near child is at the same distance as the current node
			qHeappush(qEle{minDistance, sideDistances, near})
			// # far child is further by an amount depending only
			// # on the split value
			sd := make([]float64, cx)
			copy(sd, sideDistances)
			if math.IsInf(p, 1) {
				minDistance = math.Max(minDistance, math.Abs(innernode.split-x))
			} else if p == 1 {
				sd[innernode.splitDim] = math.Abs(innernode.split - x)
				minDistance = minDistance - sideDistances[innernode.splitDim] + sd[innernode.splitDim]
			} else {
				sd[innernode.splitDim] = math.Pow(math.Abs(innernode.split-x), p)
				minDistance = minDistance - sideDistances[innernode.splitDim] + sd[innernode.splitDim]
			}
			// # far child might be too far, if so, don't bother pushing it
			if minDistance <= distanceUpperBound*epsfac {
				qHeappush(qEle{minDistance, sd, far})
			}
		}

	}
	for i := range neighbors {
		neighbors[i].float64 *= -1
		if !math.IsInf(p, 1) {
			neighbors[i].float64 = math.Pow(neighbors[i].float64, 1./p)
		}
	}
	sort.Slice(neighbors, func(i, j int) bool { return neighbors[i].float64 < neighbors[j].float64 })
	for ik := 0; ik < k && ik < len(neighbors); ik++ {
		callback(ik, neighbors[ik].float64, neighbors[ik].int)
	}

}

// Query ...
// Query the kd-tree for nearest neighbors
// Parameters
// ----------
// x : array_like, last dimension self.m
// 	An array of points to query.
// k : int, optional
// 	The number of nearest neighbors to return.
// eps : nonnegative float, optional
// 	Return approximate nearest neighbors; the kth returned value
// 	is guaranteed to be no further than (1+eps) times the
// 	distance to the real kth nearest neighbor.
// p : float, 1<=p<=infinity, optional
// 	Which Minkowski p-norm to use.
// 	1 is the sum-of-absolute-values "Manhattan" distance
// 	2 is the usual Euclidean distance
// 	infinity is the maximum-coordinate-difference distance
// distance_upper_bound : nonnegative float, optional
// 	Return only neighbors within this distance. This is used to prune
// 	tree searches, so if you are doing a series of nearest-neighbor
// 	queries, it may help to supply the distance to the nearest neighbor
// 	of the most recent point.
// Returns
// -------
// d : float or array of floats
// 	The distances to the nearest neighbors.
// 	If x has shape tuple+(self.m,), then d has shape tuple if
// 	k is one, or tuple+(k,) if k is larger than one. Missing
// 	neighbors (e.g. when k > n or distance_upper_bound is
// 	given) are indicated with infinite distances.  If k is None,
// 	then d is an object array of shape tuple, containing lists
// 	of distances. In either case the hits are sorted by distance
// 	(nearest first).
// i : integer or array of integers
// 	The locations of the neighbors in self.data. i is the same
// 	shape as d.
func (tr *KDTree) Query(X mat.Matrix, k int, eps, p, distanceUpperBound float64) (dd, ii *mat.Dense) {
	NSamples, NFeatures := X.Dims()
	dd, ii = mat.NewDense(NSamples, k, nil), mat.NewDense(NSamples, k, nil)
	base.Parallelize(runtime.NumCPU(), NSamples, func(th, start, end int) {
		XRowViewer, _ := X.(mat.RowViewer)
		var row []float64
		if XRowViewer == nil {
			row = make([]float64, NFeatures)
		}
		for sample := start; sample < end; sample++ {
			callback := func(ik int, dist float64, ind int) {
				ii.Set(sample, ik, float64(ind))
				dd.Set(sample, ik, dist)
			}
			var XRow mat.Vector
			if XRowViewer == nil {
				mat.Row(row, sample, X)
				XRow = mat.NewVecDense(NFeatures, row)
			} else {
				XRow = XRowViewer.RowView(sample)
			}
			tr._query(XRow, k, eps, p, distanceUpperBound, callback)
		}
	})
	return
}
