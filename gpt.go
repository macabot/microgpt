package microgpt

import "math/rand"

func sum(values []*Value) *Value {
	if len(values) == 0 {
		return NewValue(0, nil, nil)
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

type Layer struct {
	// AttnWq Self-Attention Query Weights
	AttnWq [][]*Value
	// AttnWk Self-Attention Key Weights
	AttnWk [][]*Value
	// AttnWv Self-Attention Value Weights
	AttnWv [][]*Value
	// AttnWo Self-Attention Output Weights
	AttnWo [][]*Value
	// MLPFC1 First Fully Connected layer of the Multilayer Perceptron
	MLPFC1 [][]*Value
	// MLPFC2 Second Fully Connected layer of the Multilayer Perceptron
	MLPFC2 [][]*Value
}

type State struct {
	HeadDim int

	// WTE Weight Token Embedding
	WTE [][]*Value
	// WPE Weight Position Embedding
	WPE [][]*Value
	// LMHead Language Modeling Head
	LMHead [][]*Value
	Layers []*Layer
}

func gauss(mean, std float32) *Value {
	data := mean + float32(rand.NormFloat64())*std
	return NewValue(data, nil, nil)
}

func matrix(nOut, nIn int, std float32) [][]*Value {
	m := make([][]*Value, nOut)
	for r := range nOut {
		row := make([]*Value, nIn)
		for i := range nIn {
			row[i] = gauss(0, std)
		}
		m[r] = row
	}
	return m
}

func NewState(vocabSize, layerSize int, embedDim int, blockSize int, heads int, std float32) *State {
	headDim := embedDim / heads
	layers := make([]*Layer, layerSize)
	for i := range layerSize {
		layers[i] = &Layer{
			AttnWq: matrix(embedDim, embedDim, std),
			AttnWk: matrix(embedDim, embedDim, std),
			AttnWv: matrix(embedDim, embedDim, std),
			AttnWo: matrix(embedDim, embedDim, std),
			MLPFC1: matrix(4*embedDim, embedDim, std),
			MLPFC2: matrix(embedDim, 4*embedDim, std),
		}
	}
	return &State{
		HeadDim: headDim,
		WTE:     matrix(vocabSize, embedDim, std),
		WPE:     matrix(blockSize, embedDim, std),
		LMHead:  matrix(vocabSize, embedDim, std),
	}
}

// func GPT(state *State, tokenID int, posID int, keys []*Value, values []*Value) []*Value {
// 	tokenEmb := state.WTE[tokenID]
// 	posEmb := state.WPE[posID]
// 	x := make([]*Value, len(tokenEmb))
// 	for i := range tokenEmb {
// 		x[i] = tokenEmb[i].Add(posEmb[i])
// 	}
// 	x = RMSNorm(x)
// 	for _, layer := range state.Layers {
// 		xResidual := x
// 		x = RMSNorm(x)
// 		q := Linear(x, layer.AttnWq)
// 		k := Linear(x, layer.AttnWk)
// 		v := Linear(x, layer.AttnWv)

// 	}

// }
