package neighbors

import (
	"fmt"
	"math"
	"runtime"
	"sort"

	"github.com/gonum/floats"
	"github.com/pa-m/sklearn/base"
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
// ----------
// x : array_like
// 	Input.
// p : float, optional
// 	Input.
func (r *Rectangle) MinDistancePoint(x []float64, p float64) float64 {
	l := len(x)
	v := make([]float64, l, l)
	for d := 0; d < l; d++ {
		v[d] = math.Max(0, math.Max(r.Mins[d]-x[d], x[d]-r.Maxes[d]))
	}
	return mat.Norm(mat.NewVecDense(l, v), p)
}

// MaxDistancePoint return the maximum distance between input and points in the hyperrectangle.
// Parameters
// ----------
// x : array_like
// 	Input array.
// p : float, optional
// 	Input.
func (r *Rectangle) MaxDistancePoint(x []float64, p float64) float64 {
	l := len(x)
	v := make([]float64, l, l)
	for d := 0; d < l; d++ {
		v[d] = math.Max(r.Maxes[d]-x[d], x[d]-r.Mins[d])
	}
	return mat.Norm(mat.NewVecDense(l, v), p)
}

// MinDistanceRectangle compute the minimum distance between points in the two hyperrectangles.
// Parameters
// ----------
// other : hyperrectangle
// 	Input.
// p : float
// 	Input.
func (r *Rectangle) MinDistanceRectangle(other *Rectangle, p float64) float64 {
	l := len(other.Maxes)
	v := make([]float64, l, l)
	for d := 0; d < l; d++ {
		v[d] = math.Max(0, math.Max(r.Mins[d]-other.Maxes[d], other.Mins[d]-r.Maxes[d]))
	}
	return mat.Norm(mat.NewVecDense(l, v), p)
}

// MaxDistanceRectangle compute the maximum distance between points in the two hyperrectangles.
// Parameters
// ----------
// other : hyperrectangle
//     Input.
// p : float, optional
//     Input.
func (r *Rectangle) MaxDistanceRectangle(other *Rectangle, p float64) float64 {
	l := len(other.Maxes)
	v := make([]float64, l, l)
	for d := 0; d < l; d++ {
		v[d] = math.Max(r.Maxes[d]-other.Mins[d], other.Maxes[d]-r.Mins[d])
	}
	return mat.Norm(mat.NewVecDense(l, v), p)
}

// KDTree for quick nearest-neighbor lookup
// This class provides an index into a set of k-dimensional points which
// can be used to rapidly look up the nearest neighbors of any point.
// Parameters
// ----------
// Data : (N,K) mat.Dense
// 	The data points to be indexed. This array is not copied, and
// 	so modifying this data will result in bogus results.
// Leafsize : int, optional
// 	The number of points at which the algorithm switches over to
// 	brute-force.  Has to be positive.
// Notes
// -----
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
	tr.Maxes = make([]float64, m, m)
	tr.Mins = make([]float64, m, m)
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
	a = make([]int, n, n)
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
	sideDistances := make([]float64, cx, cx)
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
	type qSlice []qEle

	q := []qEle{{minDistance, sideDistances, tr.Tree}}
	qLess := func(i, j int) bool { return q[i].minDistance < q[j].minDistance }
	qHeappush := func(e qEle) { q = append(q, e); sort.Slice(q, qLess) }
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
	nHeappush := func(e nEle) { neighbors = append(neighbors, e); sort.Slice(neighbors, nLess) }
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
			base.Parallelize(runtime.NumCPU(), len(idx), func(th, start, end int) {

				for iidx := start; iidx < end; iidx++ {
					fitSample := idx[iidx]
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
			})
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
			sd := make([]float64, cx, cx)
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
	for ik := 0; ik < k; ik++ {
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
			row = make([]float64, NFeatures, NFeatures)
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

/*
            elif k == 1:
                dd = np.empty(retshape,dtype=float)
                dd.fill(np.inf)
                ii = np.empty(retshape,dtype=int)
                ii.fill(self.n)
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
            for c in np.ndindex(retshape):
                hits = self.__query(x[c], k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)
                if k is None:
                    dd[c] = [d for (d,i) in hits]
                    ii[c] = [i for (d,i) in hits]
                elif k > 1:
                    for j in range(len(hits)):
                        dd[c+(j,)], ii[c+(j,)] = hits[j]
                elif k == 1:
                    if len(hits) > 0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self.__query(x, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)
            if k is None:
                return [d for (d,i) in hits], [i for (d,i) in hits]
            elif k == 1:
                if len(hits) > 0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k > 1:
                dd = np.empty(k,dtype=float)
                dd.fill(np.inf)
                ii = np.empty(k,dtype=int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")

    def __query_ball_point(self, x, r, p=2., eps=0):
        R = Rectangle(self.maxes, self.mins)

        def traverse_checking(node, rect):
            if rect.min_distance_point(x, p) > r / (1. + eps):
                return []
            elif rect.max_distance_point(x, p) < r * (1. + eps):
                return traverse_no_checking(node)
            elif isinstance(node, KDTree.leafnode):
                d = self.data[node.idx]
                return node.idx[minkowski_distance(d, x, p) <= r].tolist()
            else:
                less, greater = rect.split(node.split_dim, node.split)
                return traverse_checking(node.less, less) + \
                       traverse_checking(node.greater, greater)

        def traverse_no_checking(node):
            if isinstance(node, KDTree.leafnode):
                return node.idx.tolist()
            else:
                return traverse_no_checking(node.less) + \
                       traverse_no_checking(node.greater)

        return traverse_checking(self.tree, R)

    def query_ball_point(self, x, r, p=2., eps=0):
        """Find all points within distance r of point(s) x.
        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : positive float
            The radius of points to return.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.
        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.
        Notes
        -----
        If you have many points whose neighbors you want to find, you may save
        substantial amounts of time by putting them in a KDTree and using
        query_ball_tree.
        Examples
        --------
        >>> from scipy import spatial
        >>> x, y = np.mgrid[0:5, 0:5]
        >>> points = np.c_[x.ravel(), y.ravel()]
        >>> tree = spatial.KDTree(points)
        >>> tree.query_ball_point([2, 0], 1)
        [5, 10, 11, 15]
        Query multiple points and plot the results:
        >>> import matplotlib.pyplot as plt
        >>> points = np.asarray(points)
        >>> plt.plot(points[:,0], points[:,1], '.')
        >>> for results in tree.query_ball_point(([2, 0], [3, 3]), 1):
        ...     nearby_points = points[results]
        ...     plt.plot(nearby_points[:,0], nearby_points[:,1], 'o')
        >>> plt.margins(0.1, 0.1)
        >>> plt.show()
        """
        x = np.asarray(x)
        if x.shape[-1] != self.m:
            raise ValueError("Searching for a %d-dimensional point in a "
                             "%d-dimensional KDTree" % (x.shape[-1], self.m))
        if len(x.shape) == 1:
            return self.__query_ball_point(x, r, p, eps)
        else:
            retshape = x.shape[:-1]
            result = np.empty(retshape, dtype=object)
            for c in np.ndindex(retshape):
                result[c] = self.__query_ball_point(x[c], r, p=p, eps=eps)
            return result

    def query_ball_tree(self, other, r, p=2., eps=0):
        """Find all pairs of points whose distance is at most r
        Parameters
        ----------
        other : KDTree instance
            The tree containing points to search against.
        r : float
            The maximum distance, has to be positive.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        Returns
        -------
        results : list of lists
            For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            list of the indices of its neighbors in ``other.data``.
        """
        results = [[] for i in range(self.n)]

        def traverse_checking(node1, rect1, node2, rect2):
            if rect1.min_distance_rectangle(rect2, p) > r/(1.+eps):
                return
            elif rect1.max_distance_rectangle(rect2, p) < r*(1.+eps):
                traverse_no_checking(node1, node2)
            elif isinstance(node1, KDTree.leafnode):
                if isinstance(node2, KDTree.leafnode):
                    d = other.data[node2.idx]
                    for i in node1.idx:
                        results[i] += node2.idx[minkowski_distance(d,self.data[i],p) <= r].tolist()
                else:
                    less, greater = rect2.split(node2.split_dim, node2.split)
                    traverse_checking(node1,rect1,node2.less,less)
                    traverse_checking(node1,rect1,node2.greater,greater)
            elif isinstance(node2, KDTree.leafnode):
                less, greater = rect1.split(node1.split_dim, node1.split)
                traverse_checking(node1.less,less,node2,rect2)
                traverse_checking(node1.greater,greater,node2,rect2)
            else:
                less1, greater1 = rect1.split(node1.split_dim, node1.split)
                less2, greater2 = rect2.split(node2.split_dim, node2.split)
                traverse_checking(node1.less,less1,node2.less,less2)
                traverse_checking(node1.less,less1,node2.greater,greater2)
                traverse_checking(node1.greater,greater1,node2.less,less2)
                traverse_checking(node1.greater,greater1,node2.greater,greater2)

        def traverse_no_checking(node1, node2):
            if isinstance(node1, KDTree.leafnode):
                if isinstance(node2, KDTree.leafnode):
                    for i in node1.idx:
                        results[i] += node2.idx.tolist()
                else:
                    traverse_no_checking(node1, node2.less)
                    traverse_no_checking(node1, node2.greater)
            else:
                traverse_no_checking(node1.less, node2)
                traverse_no_checking(node1.greater, node2)

        traverse_checking(self.tree, Rectangle(self.maxes, self.mins),
                          other.tree, Rectangle(other.maxes, other.mins))
        return results

    def query_pairs(self, r, p=2., eps=0):
        """
        Find all pairs of points within a distance.
        Parameters
        ----------
        r : positive float
            The maximum distance.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        Returns
        -------
        results : set
            Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding
            positions are close.
        """
        results = set()

        def traverse_checking(node1, rect1, node2, rect2):
            if rect1.min_distance_rectangle(rect2, p) > r/(1.+eps):
                return
            elif rect1.max_distance_rectangle(rect2, p) < r*(1.+eps):
                traverse_no_checking(node1, node2)
            elif isinstance(node1, KDTree.leafnode):
                if isinstance(node2, KDTree.leafnode):
                    # Special care to avoid duplicate pairs
                    if id(node1) == id(node2):
                        d = self.data[node2.idx]
                        for i in node1.idx:
                            for j in node2.idx[minkowski_distance(d,self.data[i],p) <= r]:
                                if i < j:
                                    results.add((i,j))
                    else:
                        d = self.data[node2.idx]
                        for i in node1.idx:
                            for j in node2.idx[minkowski_distance(d,self.data[i],p) <= r]:
                                if i < j:
                                    results.add((i,j))
                                elif j < i:
                                    results.add((j,i))
                else:
                    less, greater = rect2.split(node2.split_dim, node2.split)
                    traverse_checking(node1,rect1,node2.less,less)
                    traverse_checking(node1,rect1,node2.greater,greater)
            elif isinstance(node2, KDTree.leafnode):
                less, greater = rect1.split(node1.split_dim, node1.split)
                traverse_checking(node1.less,less,node2,rect2)
                traverse_checking(node1.greater,greater,node2,rect2)
            else:
                less1, greater1 = rect1.split(node1.split_dim, node1.split)
                less2, greater2 = rect2.split(node2.split_dim, node2.split)
                traverse_checking(node1.less,less1,node2.less,less2)
                traverse_checking(node1.less,less1,node2.greater,greater2)

                # Avoid traversing (node1.less, node2.greater) and
                # (node1.greater, node2.less) (it's the same node pair twice
                # over, which is the source of the complication in the
                # original KDTree.query_pairs)
                if id(node1) != id(node2):
                    traverse_checking(node1.greater,greater1,node2.less,less2)

                traverse_checking(node1.greater,greater1,node2.greater,greater2)

        def traverse_no_checking(node1, node2):
            if isinstance(node1, KDTree.leafnode):
                if isinstance(node2, KDTree.leafnode):
                    # Special care to avoid duplicate pairs
                    if id(node1) == id(node2):
                        for i in node1.idx:
                            for j in node2.idx:
                                if i < j:
                                    results.add((i,j))
                    else:
                        for i in node1.idx:
                            for j in node2.idx:
                                if i < j:
                                    results.add((i,j))
                                elif j < i:
                                    results.add((j,i))
                else:
                    traverse_no_checking(node1, node2.less)
                    traverse_no_checking(node1, node2.greater)
            else:
                # Avoid traversing (node1.less, node2.greater) and
                # (node1.greater, node2.less) (it's the same node pair twice
                # over, which is the source of the complication in the
                # original KDTree.query_pairs)
                if id(node1) == id(node2):
                    traverse_no_checking(node1.less, node2.less)
                    traverse_no_checking(node1.less, node2.greater)
                    traverse_no_checking(node1.greater, node2.greater)
                else:
                    traverse_no_checking(node1.less, node2)
                    traverse_no_checking(node1.greater, node2)

        traverse_checking(self.tree, Rectangle(self.maxes, self.mins),
                          self.tree, Rectangle(self.maxes, self.mins))
        return results

    def count_neighbors(self, other, r, p=2.):
        """
        Count how many nearby pairs can be formed.
        Count the number of pairs (x1,x2) can be formed, with x1 drawn
        from self and x2 drawn from `other`, and where
        ``distance(x1, x2, p) <= r``.
        This is the "two-point correlation" described in Gray and Moore 2000,
        "N-body problems in statistical learning", and the code here is based
        on their algorithm.
        Parameters
        ----------
        other : KDTree instance
            The other tree to draw points from.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched with
            a single tree traversal.
        p : float, 1<=p<=infinity, optional
            Which Minkowski p-norm to use
        Returns
        -------
        result : int or 1-D array of ints
            The number of pairs. Note that this is internally stored in a numpy
            int, and so may overflow if very large (2e9).
        """
        def traverse(node1, rect1, node2, rect2, idx):
            min_r = rect1.min_distance_rectangle(rect2,p)
            max_r = rect1.max_distance_rectangle(rect2,p)
            c_greater = r[idx] > max_r
            result[idx[c_greater]] += node1.children*node2.children
            idx = idx[(min_r <= r[idx]) & (r[idx] <= max_r)]
            if len(idx) == 0:
                return

            if isinstance(node1,KDTree.leafnode):
                if isinstance(node2,KDTree.leafnode):
                    ds = minkowski_distance(self.data[node1.idx][:,np.newaxis,:],
                                  other.data[node2.idx][np.newaxis,:,:],
                                  p).ravel()
                    ds.sort()
                    result[idx] += np.searchsorted(ds,r[idx],side='right')
                else:
                    less, greater = rect2.split(node2.split_dim, node2.split)
                    traverse(node1, rect1, node2.less, less, idx)
                    traverse(node1, rect1, node2.greater, greater, idx)
            else:
                if isinstance(node2,KDTree.leafnode):
                    less, greater = rect1.split(node1.split_dim, node1.split)
                    traverse(node1.less, less, node2, rect2, idx)
                    traverse(node1.greater, greater, node2, rect2, idx)
                else:
                    less1, greater1 = rect1.split(node1.split_dim, node1.split)
                    less2, greater2 = rect2.split(node2.split_dim, node2.split)
                    traverse(node1.less,less1,node2.less,less2,idx)
                    traverse(node1.less,less1,node2.greater,greater2,idx)
                    traverse(node1.greater,greater1,node2.less,less2,idx)
                    traverse(node1.greater,greater1,node2.greater,greater2,idx)

        R1 = Rectangle(self.maxes, self.mins)
        R2 = Rectangle(other.maxes, other.mins)
        if np.shape(r) == ():
            r = np.array([r])
            result = np.zeros(1,dtype=int)
            traverse(self.tree, R1, other.tree, R2, np.arange(1))
            return result[0]
        elif len(np.shape(r)) == 1:
            r = np.asarray(r)
            n, = r.shape
            result = np.zeros(n,dtype=int)
            traverse(self.tree, R1, other.tree, R2, np.arange(n))
            return result
        else:
            raise ValueError("r must be either a single value or a one-dimensional array of values")

    def sparse_distance_matrix(self, other, max_distance, p=2.):
        """
        Compute a sparse distance matrix
        Computes a distance matrix between two KDTrees, leaving as zero
        any distance greater than max_distance.
        Parameters
        ----------
        other : KDTree
        max_distance : positive float
        p : float, optional
        Returns
        -------
        result : dok_matrix
            Sparse matrix representing the results in "dictionary of keys" format.
        """
        result = scipy.sparse.dok_matrix((self.n,other.n))

        def traverse(node1, rect1, node2, rect2):
            if rect1.min_distance_rectangle(rect2, p) > max_distance:
                return
            elif isinstance(node1, KDTree.leafnode):
                if isinstance(node2, KDTree.leafnode):
                    for i in node1.idx:
                        for j in node2.idx:
                            d = minkowski_distance(self.data[i],other.data[j],p)
                            if d <= max_distance:
                                result[i,j] = d
                else:
                    less, greater = rect2.split(node2.split_dim, node2.split)
                    traverse(node1,rect1,node2.less,less)
                    traverse(node1,rect1,node2.greater,greater)
            elif isinstance(node2, KDTree.leafnode):
                less, greater = rect1.split(node1.split_dim, node1.split)
                traverse(node1.less,less,node2,rect2)
                traverse(node1.greater,greater,node2,rect2)
            else:
                less1, greater1 = rect1.split(node1.split_dim, node1.split)
                less2, greater2 = rect2.split(node2.split_dim, node2.split)
                traverse(node1.less,less1,node2.less,less2)
                traverse(node1.less,less1,node2.greater,greater2)
                traverse(node1.greater,greater1,node2.less,less2)
                traverse(node1.greater,greater1,node2.greater,greater2)
        traverse(self.tree, Rectangle(self.maxes, self.mins),
                 other.tree, Rectangle(other.maxes, other.mins))

        return result


def distance_matrix(x, y, p=2, threshold=1000000):
    """
    Compute the distance matrix.
    Returns the matrix of all pair-wise distances.
    Parameters
    ----------
    x : (M, K) array_like
        Matrix of M vectors in K dimensions.
    y : (N, K) array_like
        Matrix of N vectors in K dimensions.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    threshold : positive int
        If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead
        of large temporary arrays.
    Returns
    -------
    result : (M, N) ndarray
        Matrix containing the distance from every vector in `x` to every vector
        in `y`.
    Examples
    --------
    >>> from scipy.spatial import distance_matrix
    >>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])
    array([[ 1.        ,  1.41421356],
           [ 1.41421356,  1.        ]])
    """

    x = np.asarray(x)
    m, k = x.shape
    y = np.asarray(y)
    n, kk = y.shape

    if k != kk:
        raise ValueError("x contains %d-dimensional vectors but y contains %d-dimensional vectors" % (k, kk))

    if m*n*k <= threshold:
        return minkowski_distance(x[:,np.newaxis,:],y[np.newaxis,:,:],p)
    else:
        result = np.empty((m,n),dtype=float)  # FIXME: figure out the best dtype
        if m < n:
            for i in range(m):
                result[i,:] = minkowski_distance(x[i],y,p)
        else:
            for j in range(n):
                result[:,j] = minkowski_distance(x,y[j],p)
        return result
*/
/*
// TreeNode used for KdTree
type TreeNode struct {
	Lower, Higher *TreeNode
	Index         int
}

// NChildren return count of children, not including self
func (node *TreeNode) NChildren() int {
	n := 0
	if node.Lower != nil {
		n += 1 + node.Lower.NChildren()
	}
	if node.Higher != nil {
		n += 1 + node.Higher.NChildren()
	}
	return n
}
func (node *TreeNode) allIndices() []int {
	a := make([]int, 0)
	if node.Lower != nil {
		a = append(a, node.Lower.allIndices()...)
	}
	a = append(a, node.Index)
	if node.Higher != nil {
		a = append(a, node.Higher.allIndices()...)
	}
	return a
}

func buildkdtree(X mat.Matrix, samplesIndex []int, depth int) *TreeNode {
	_, k := X.Dims()
	// v https://fr.wikipedia.org/wiki/Arbre_kd
	if len(samplesIndex) == 0 {
		return nil
	}
	axis := depth % k
	// find index of median in samplesIndex
	idx := make([]int, len(samplesIndex))
	for iidx := range idx {
		idx[iidx] = iidx
	}
	sort.Slice(idx, func(i, j int) bool {
		return X.At(samplesIndex[idx[i]], axis) < X.At(samplesIndex[idx[j]], axis)
	})
	samplesIndexMedian := samplesIndex[idx[len(idx)/2]]
	L, R := make([]int, 0), make([]int, 0)

	for _, ix := range samplesIndex {
		if ix != samplesIndexMedian {
			if X.At(ix, axis) < X.At(samplesIndexMedian, axis) {
				L = append(L, ix)
			} else {
				R = append(R, ix)
			}
		}
	}
	node := &TreeNode{Index: samplesIndexMedian}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { node.Lower = buildkdtree(X, L, depth+1); wg.Done() }()
	go func() { node.Higher = buildkdtree(X, R, depth+1); wg.Done() }()
	wg.Wait()
	return node
}

// KdTree struct
type KdTree struct {
	Root     *TreeNode
	X        *mat.Dense
	Distance Distance
}

// NewKdTree returns an initialized *KdTree
func NewKdTree(X mat.Matrix, leafsize int, metrics string) *KdTree {
	NSamples, _ := X.Dims()
	samplesIndex := make([]int, NSamples, NSamples)
	for i := range samplesIndex {
		samplesIndex[i] = i
	}
	var dist Distance
	switch metrics {
	case "manhattan":
		dist = MinkowskiDistance(1)
	default:
		dist = MinkowskiDistance(2)
	}
	return &KdTree{
		Root:     buildkdtree(X, samplesIndex, 0),
		X:        mat.DenseCopyOf(X),
		Distance: dist,
	}
}

// Query returns K nearest distances and indices
func (tree *KdTree) Query(X mat.Dense, K int) (distances, indices *mat.Dense) {
	NSamples, NFeatures := X.Dims()
	distances = mat.NewDense(NSamples, K, nil)
	indices = mat.NewDense(NSamples, K, nil)
	querySample := func(node *TreeNode, sample, depth int) []int { return []int{} }
	querySample = func(node *TreeNode, sample, depth int) []int {
		if node == nil {
			return []int{}
		}
		axis := depth % NFeatures
		var a []int
		if X.At(sample, axis) < tree.X.At(node.Index, axis) {
			a := querySample(node.Lower, sample, depth+1)
			if len(a) < K {
				a = append(a, node.Index)
			}
			if len(a) < K {
				a = append(a, querySample(node.Higher, sample, depth+1)...)
			}
		} else {
			a := querySample(node.Higher, sample, depth+1)
			if len(a) < K {
				a = append(a, node.Index)
			}
			if len(a) < K {
				a = append(a, querySample(node.Lower, sample, depth+1)...)
			}
		}
		sort.Slice(a, func(i, j int) bool {
			return tree.Distance(X.RowView(sample), tree.X.RowView(a[i])) < tree.Distance(X.RowView(sample), tree.X.RowView(a[j]))
		})
		if len(a) > K {
			a = a[0:K]
		}
		return a
	}

	base.Parallelize(runtime.NumCPU(), NSamples, func(th, start, end int) {
		for sample := start; sample < end; sample++ {
			a := querySample(tree.Root, sample, 0)
			for k := range a {
				indices.Set(sample, k, float64(a[k]))
				distances.Set(sample, k, tree.Distance(X.RowView(sample), tree.X.RowView(a[k])))
			}
		}
	})
	return
}
*/
