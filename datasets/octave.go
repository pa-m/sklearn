package datasets

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// LoadOctaveBin reads an (possibly gzipped) octave binary file into a map of *map.Dense
func LoadOctaveBin(filename string) map[string]*mat.Dense {
	retval := make(map[string]*mat.Dense)
	// v https://lists.gnu.org/archive/html/help-octave/2004-11/msg00068.html
	var f0 *os.File
	var err error
	var b []byte
	type PosReader struct {
		io.ReadCloser
		Pos int
	}
	read := func(f *PosReader, n int) ([]byte, error) {
		b := make([]byte, n)
		nread, err := f.ReadCloser.Read(b)
		if err != nil {
			return b, err
		}
		f.Pos += nread
		if nread != n {
			panic(fmt.Errorf("%d/%d bytes read", nread, n))
		}
		if err != nil {
			panic(err)
		}
		return b, nil
	}
	readUint32 := func(f *PosReader) (uint32, error) {

		b, err := read(f, 4)
		if err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint32(b[0:]), nil
	}
	readString := func(f *PosReader) (string, error) {

		lenString, err := readUint32(f)
		if err != nil {
			return "", err
		}
		b, err := read(f, int(lenString))
		if err != nil {
			return "", err
		}
		return string(b), nil
	}
	f0, err = os.Open(filename)
	check(err)
	defer f0.Close()

	var f *PosReader
	if strings.HasSuffix(filename, ".gz") {
		reader, err := gzip.NewReader(f0)
		check(err)
		f = &PosReader{ReadCloser: reader}

	} else {
		f = &PosReader{ReadCloser: f}
	}

	magic := "Octave-1-L"
	b, err = read(f, 10)
	check(err)
	if string(b) != magic {
		panic("not a octave binary file")
	}

	_, err = read(f, 1)
	check(err)
	for {
		//fmt.Println("Pos:", f.Pos)
		var VarName string
		VarName, err = readString(f)
		if err == io.EOF {
			return retval
		}
		check(err)
		//fmt.Printf("varname:%s\n", VarName)
		//read doclength (4)
		read(f, 4)
		//read doc (1 byte)
		read(f, 1)
		// read datatype
		b, err = read(f, 1)
		check(err)
		if b[0] != 0xff {
			panic("0xff expected")
		}
		var datatype string
		datatype, err = readString(f)
		check(err)
		if datatype != "matrix" {
			panic("matrix expected")
		}
		// read FE FF FF FF
		read(f, 4)
		var Rows, Cols uint32
		Rows, err = readUint32(f)
		check(err)
		Cols, err = readUint32(f)
		check(err)
		//fmt.Printf("%d x %d\n", Rows, Cols)
		// read 1 unknown byte
		read(f, 1)
		data := make([]float64, Rows*Cols, Rows*Cols)
		err = binary.Read(f, binary.LittleEndian, data)
		f.Pos += int(Rows * Cols * 8)
		check(err)
		// transpose:

		t := mat.NewDense(int(Cols), int(Rows), data)
		retval[VarName] = mat.DenseCopyOf(t.T())
	}

}
