package microgpt

func sum(values []*Value) *Value {
	if len(values) == 0 {
		return nil
	}
	s := values[0]
	for i := 1; i < len(values); i++ {
		s = s.Add(values[i])
	}
	return s
}

func Linear(x []*Value, w [][]*Value) []*Value {
	out := make([]*Value, len(w))
	for r, row := range w {
		weighted := make([]*Value, len(x))
		for i, v := range x {
			weight := row[i]
			weighted[i] = weight.Mul(v)
		}
		out[r] = sum(weighted)
	}
	return out
}

func max(values []*Value) *Value {
	m := values[0]
	for i := 1; i < len(values); i++ {
		if m.Data < values[i].Data {
			m = values[i]
		}
	}
	return m
}

func Softmax(logits []*Value) []*Value {
	maxVal := max(logits)
	exps := make([]*Value, len(logits))
	for i, val := range logits {
		exps[i] = val.Sub(maxVal).Exp()
	}
	total := sum(exps)
	sm := make([]*Value, len(exps))
	for i, e := range exps {
		sm[i] = e.Div(total)
	}
	return sm
}

// RMSNorm perform Root Mean Square Normalization
func RMSNorm(x []*Value) []*Value {
	squares := make([]*Value, len(x))
	for i, v := range x {
		squares[i] = v.Mul(v)
	}
	ms := sum(squares).Div(NewValue(float32(len(x)), nil, nil))
	epsilon := NewValue(1e-5, nil, nil)
	scale := ms.Add(epsilon).Pow(-0.5)
	norm := make([]*Value, len(x))
	for i, v := range x {
		norm[i] = v.Mul(scale)
	}
	return norm
}
