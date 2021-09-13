package nnga

import (
	"errors"

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
		relu(ret)
	}
	return ret, nil
}

func relu(m *gmatrix.Matrix) (*gmatrix.Matrix, error) {
	fn := func(val float64) (float64, error) {
		if val > 0 {
			return val, nil
		} else {
			return 0, nil
		}
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
