package nnga

import (
	"bufio"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/TTRSQ/gmatrix"
)

type NNGA struct {
	tensors []*gmatrix.Matrix
}

func NewNNGA(tensors []*gmatrix.Matrix) (*NNGA, error) {
	if len(tensors) < 1 {
		return nil, errors.New("tensors is empty")
	}
	if len(tensors) >= 2 {
		for i := 1; i < len(tensors); i++ {
			if tensors[i-1].C() != tensors[i].R() {
				return nil, errors.New("invalid shape")
			}
		}
	}
	return &NNGA{
		tensors: tensors,
	}, nil
}

func (nnga *NNGA) Forward(input []float64) (*gmatrix.Matrix, error) {
	ret, err := gmatrix.NewMatrix(1, len(input), input)
	if err != nil {
		return nil, err
	}
	for i := range nnga.tensors {
		ret, err = ret.Mul(nnga.tensors[i])
		if err != nil {
			return nil, err
		}
		ret, err = relu(ret)
		if err != nil {
			return nil, err
		}
	}
	return ret, nil
}

func (nnga *NNGA) ParallelForward(input []float64) (*gmatrix.Matrix, error) {
	ret, err := gmatrix.NewMatrix(1, len(input), input)
	if err != nil {
		return nil, err
	}
	for i := range nnga.tensors {
		ret, err = ret.MulParallel(nnga.tensors[i])
		if err != nil {
			return nil, err
		}
		ret, err = relu(ret)
		if err != nil {
			return nil, err
		}
	}
	return ret, nil
}

func (nnga *NNGA) ForwardSig(input []float64) (*gmatrix.Matrix, error) {
	ret, err := gmatrix.NewMatrix(1, len(input), input)
	if err != nil {
		return nil, err
	}
	for i := range nnga.tensors {
		ret, err = ret.Mul(nnga.tensors[i])
		if err != nil {
			return nil, err
		}
		ret, err = sigmoid(ret)
		if err != nil {
			return nil, err
		}
	}
	return ret, nil
}

func sigmoid(m *gmatrix.Matrix) (*gmatrix.Matrix, error) {
	fn := func(val float64) (float64, error) {
		return 1 / (1 + math.Exp(-val)), nil
	}
	return m.Func(fn)
}

func relu(m *gmatrix.Matrix) (*gmatrix.Matrix, error) {
	fn := func(val float64) (float64, error) {
		return math.Max(0, val), nil
	}
	return m.Func(fn)
}

func (nngaa *NNGA) Marge(nngab *NNGA, orgRate float64) (*NNGA, error) {
	tensors := []*gmatrix.Matrix{}
	if len(nngaa.tensors) != len(nngab.tensors) {
		return nil, errors.New("different tensor size")
	}
	for t := range nngaa.tensors {
		mat, err := nngaa.tensors[t].RandMerge(nngab.tensors[t], orgRate)
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, mat)
	}
	return NewNNGA(tensors)
}

func (nngaa *NNGA) Mean(nngab *NNGA) (*NNGA, error) {
	tensors := []*gmatrix.Matrix{}
	if len(nngaa.tensors) != len(nngab.tensors) {
		return nil, errors.New("different tensor size")
	}
	for t := range nngaa.tensors {
		mat, err := nngaa.tensors[t].Mean(nngab.tensors[t])
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, mat)
	}
	return NewNNGA(tensors)
}

func (nnga *NNGA) Save(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()
	for i := range nnga.tensors {
		r := nnga.tensors[i].R()
		c := nnga.tensors[i].C()
		_, err = file.Write(([]byte)(fmt.Sprintf("%d %d,", r, c)))
		if err != nil {
			return err
		}
		dataStr := ""
		for j := range nnga.tensors[i].Datas() {
			if j != 0 {
				dataStr += " "
			}
			dataStr += fmt.Sprint(nnga.tensors[i].Datas()[j])
		}
		_, err = file.Write(([]byte)(fmt.Sprintf("%s\n", dataStr)))
		if err != nil {
			return err
		}
	}
	return nil
}

func (nnga *NNGA) Load(filePath string) (*NNGA, error) {
	fp, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer fp.Close()
	scanner := bufio.NewScanner(fp)

	tensors := []*gmatrix.Matrix{}
	for scanner.Scan() {
		row := strings.Split(scanner.Text(), ",")
		head := strings.Split(row[0], " ")
		r, err := strconv.ParseInt(head[0], 10, 64)
		if err != nil {
			return nil, err
		}
		c, err := strconv.ParseInt(head[1], 10, 64)
		if err != nil {
			return nil, err
		}
		data := strings.Split(row[1], " ")
		datas := []float64{}
		for i := range data {
			f, err := strconv.ParseFloat(data[i], 64)
			if err != nil {
				return nil, err
			}
			datas = append(datas, f)
		}
		mat, err := gmatrix.NewMatrix(int(r), int(c), datas)
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, mat)
	}

	if err = scanner.Err(); err != nil {
		return nil, err
	}

	return NewNNGA(tensors)
}
