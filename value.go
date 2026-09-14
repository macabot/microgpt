package microgpt

import "math"

type Value struct {
	Data       float32
	Grad       float32
	Children   []*Value
	LocalGrads []float32
}

func NewValue(data float32, children []*Value, localGrads []float32) *Value {
	return &Value{
		Data:       data,
		Grad:       0,
		Children:   children,
		LocalGrads: localGrads,
	}
}

func (v *Value) Add(other *Value) *Value {
	return NewValue(
		v.Data+other.Data,
		[]*Value{v, other},
		[]float32{1, 1},
	)
}

func (v *Value) Sub(other *Value) *Value {
	return NewValue(
		v.Data-other.Data,
		[]*Value{v, other},
		[]float32{1, -1},
	)
}

func (v *Value) Mul(other *Value) *Value {
	return NewValue(
		v.Data*other.Data,
		[]*Value{v, other},
		[]float32{other.Data, v.Data},
	)
}

func (v *Value) Div(other *Value) *Value {
	return NewValue(
		v.Data/other.Data,
		[]*Value{v, other},
		[]float32{1 / other.Data, -v.Data / (other.Data * other.Data)},
	)
}

func pow32(base, exponent float32) float32 {
	return float32(math.Pow(float64(base), float64(exponent)))
}

func (v *Value) Pow(other float32) *Value {
	return NewValue(
		pow32(v.Data, other),
		[]*Value{v},
		[]float32{other * pow32(v.Data, other-1)},
	)
}

func (v *Value) Log() *Value {
	return NewValue(
		float32(math.Log(float64(v.Data))),
		[]*Value{v},
		[]float32{1 / v.Data},
	)
}

func (v *Value) Exp() *Value {
	e := float32(math.Exp(float64(v.Data)))
	return NewValue(
		e,
		[]*Value{v},
		[]float32{e},
	)
}

func (v *Value) ReLU() *Value {
	data := v.Data
	var localGrad float32 = 1.0
	if data <= 0 {
		data = 0
		localGrad = 0
	}
	return NewValue(
		data,
		[]*Value{v},
		[]float32{localGrad},
	)
}

func (v *Value) backward() {
	topo := []*Value{}
	visited := map[*Value]struct{}{}
	var buildTopo func(v *Value)
	buildTopo = func(v *Value) {
		if _, ok := visited[v]; !ok {
			visited[v] = struct{}{}
			for _, child := range v.Children {
				buildTopo(child)
			}
			topo = append(topo, v)
		}
	}
	buildTopo(v)
	v.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		for j := range v.Children {
			child := v.Children[j]
			localGrad := v.LocalGrads[j]
			child.Grad += localGrad * v.Grad
		}
	}
}
