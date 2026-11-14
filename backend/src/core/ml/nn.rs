use crate::core::ml::algorithms::MlError;

/// Activation functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Identity,
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            } // LeakyReLU for stability
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Identity => x,
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::Identity => 1.0,
        }
    }
}

/// Simple PRNG (xorshift64) with deterministic seed for reproducibility
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    pub fn next_f64(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let v = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64) / ((1_u64 << 53) as f64)
    }

    pub fn next_uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }
}

// -------------------------------------------------------------
// Multi-Layer Perceptron (MLP)
// -------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Vec<Vec<f64>>, // [out_features][in_features]
    pub biases: Vec<f64>,       // [out_features]
    pub activation: Activation,
}

impl DenseLayer {
    pub fn new(in_features: usize, out_features: usize, activation: Activation, rng: &mut Rng) -> Self {
        // Xavier/He initialization
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();
        let mut weights = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            let mut row = Vec::with_capacity(in_features);
            for _ in 0..in_features {
                row.push(rng.next_uniform(-limit, limit));
            }
            weights.push(row);
        }
        let biases = vec![0.0; out_features];
        Self {
            in_features,
            out_features,
            weights,
            biases,
            activation,
        }
    }

    pub fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut pre_act = Vec::with_capacity(self.out_features);
        let mut post_act = Vec::with_capacity(self.out_features);
        for i in 0..self.out_features {
            let mut sum = self.biases[i];
            for j in 0..self.in_features {
                sum += self.weights[i][j] * input[j];
            }
            pre_act.push(sum);
            post_act.push(self.activation.apply(sum));
        }
        (pre_act, post_act)
    }
}

#[derive(Debug, Clone)]
pub struct MlpNetwork {
    pub layers: Vec<DenseLayer>,
}

impl MlpNetwork {
    pub fn new(layer_sizes: &[usize], activations: &[Activation], seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut layers = Vec::new();
        for i in 0..(layer_sizes.len() - 1) {
            layers.push(DenseLayer::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                activations[i],
                &mut rng,
            ));
        }
        Self { layers }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            let (_, post) = layer.forward(&current);
            current = post;
        }
        current
    }

    pub fn train_classification(
        &mut self,
        x: &[Vec<f64>],
        y_indices: &[usize],
        _num_classes: usize,
        epochs: usize,
        learning_rate: f64,
    ) {
        let n_samples = x.len();
        if n_samples == 0 {
            return;
        }

        for _ in 0..epochs {
            for (sample_x, &target_class) in x.iter().zip(y_indices.iter()) {
                // Forward pass with activations stored
                let mut layer_inputs = Vec::with_capacity(self.layers.len() + 1);
                let mut layer_pre_acts = Vec::with_capacity(self.layers.len());
                let mut current = sample_x.clone();
                layer_inputs.push(current.clone());

                for layer in &self.layers {
                    let (pre, post) = layer.forward(&current);
                    layer_pre_acts.push(pre);
                    current = post;
                    layer_inputs.push(current.clone());
                }

                // Softmax on output layer
                let output = current;
                let max_val = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_scores: Vec<f64> = output.iter().map(|v| (v - max_val).exp()).collect();
                let sum_exp: f64 = exp_scores.iter().sum::<f64>().max(1e-15);
                let probs: Vec<f64> = exp_scores.iter().map(|v| v / sum_exp).collect();

                // Cross-entropy gradient for softmax output: probs - one_hot
                let mut delta = probs;
                if target_class < delta.len() {
                    delta[target_class] -= 1.0;
                }

                // Backpropagate
                for l in (0..self.layers.len()).rev() {
                    let in_act = &layer_inputs[l];
                    let mut prev_delta = vec![0.0; self.layers[l].in_features];

                    for out_idx in 0..self.layers[l].out_features {
                        let d = delta[out_idx];
                        self.layers[l].biases[out_idx] -= learning_rate * d;

                        for in_idx in 0..self.layers[l].in_features {
                            prev_delta[in_idx] += self.layers[l].weights[out_idx][in_idx] * d;
                            self.layers[l].weights[out_idx][in_idx] -= learning_rate * d * in_act[in_idx];
                        }
                    }

                    if l > 0 {
                        let pre_acts = &layer_pre_acts[l - 1];
                        for in_idx in 0..self.layers[l].in_features {
                            prev_delta[in_idx] *= self.layers[l - 1].activation.derivative(pre_acts[in_idx]);
                        }
                        delta = prev_delta;
                    }
                }
            }
        }
    }

    pub fn train_regression(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        epochs: usize,
        learning_rate: f64,
    ) {
        let n_samples = x.len();
        if n_samples == 0 {
            return;
        }

        for _ in 0..epochs {
            for (sample_x, &target_val) in x.iter().zip(y.iter()) {
                let mut layer_inputs = Vec::with_capacity(self.layers.len() + 1);
                let mut layer_pre_acts = Vec::with_capacity(self.layers.len());
                let mut current = sample_x.clone();
                layer_inputs.push(current.clone());

                for layer in &self.layers {
                    let (pre, post) = layer.forward(&current);
                    layer_pre_acts.push(pre);
                    current = post;
                    layer_inputs.push(current.clone());
                }

                // MSE loss: output - target
                let pred = current[0];
                let mut delta = vec![pred - target_val];

                for l in (0..self.layers.len()).rev() {
                    let in_act = &layer_inputs[l];
                    let mut prev_delta = vec![0.0; self.layers[l].in_features];

                    for out_idx in 0..self.layers[l].out_features {
                        let d = delta[out_idx];
                        self.layers[l].biases[out_idx] -= learning_rate * d;

                        for in_idx in 0..self.layers[l].in_features {
                            prev_delta[in_idx] += self.layers[l].weights[out_idx][in_idx] * d;
                            self.layers[l].weights[out_idx][in_idx] -= learning_rate * d * in_act[in_idx];
                        }
                    }

                    if l > 0 {
                        let pre_acts = &layer_pre_acts[l - 1];
                        for in_idx in 0..self.layers[l].in_features {
                            prev_delta[in_idx] *= self.layers[l - 1].activation.derivative(pre_acts[in_idx]);
                        }
                        delta = prev_delta;
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------
// 1D Convolutional Neural Network (CNN)
// -------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Conv1dLayer {
    pub in_channels: usize,
    pub num_filters: usize,
    pub kernel_size: usize,
    pub kernels: Vec<Vec<Vec<f64>>>, // [filter][channel][k]
    pub biases: Vec<f64>,            // [filter]
}

impl Conv1dLayer {
    pub fn new(in_channels: usize, num_filters: usize, kernel_size: usize, rng: &mut Rng) -> Self {
        let limit = (6.0 / (kernel_size * (in_channels + num_filters)) as f64).sqrt();
        let mut kernels = Vec::with_capacity(num_filters);
        for _ in 0..num_filters {
            let mut filter = Vec::with_capacity(in_channels);
            for _ in 0..in_channels {
                let mut k_weights = Vec::with_capacity(kernel_size);
                for _ in 0..kernel_size {
                    k_weights.push(rng.next_uniform(-limit, limit));
                }
                filter.push(k_weights);
            }
            kernels.push(filter);
        }
        let biases = vec![0.0; num_filters];
        Self {
            in_channels,
            num_filters,
            kernel_size,
            kernels,
            biases,
        }
    }

    /// Forward pass for a 1D sequence of length L: input [in_channels][L] -> output [num_filters][L]
    pub fn forward(&self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let length = input[0].len();
        let half_k = (self.kernel_size / 2) as isize;
        let mut out = vec![vec![0.0; length]; self.num_filters];

        for f in 0..self.num_filters {
            for i in 0..length {
                let mut sum = self.biases[f];
                for c in 0..self.in_channels {
                    for k in 0..self.kernel_size {
                        let in_idx = (i as isize) + (k as isize) - half_k;
                        let val = if in_idx >= 0 && (in_idx as usize) < length {
                            input[c][in_idx as usize]
                        } else {
                            0.0
                        };
                        sum += self.kernels[f][c][k] * val;
                    }
                }
                // LeakyReLU
                out[f][i] = if sum > 0.0 { sum } else { 0.01 * sum };
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct CnnModel {
    pub conv: Conv1dLayer,
    pub dense1: DenseLayer,
    pub dense_out: DenseLayer,
}

impl CnnModel {
    pub fn new(num_features: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let num_filters = 8;
        let kernel_size = if num_features >= 3 { 3 } else { num_features.max(1) };
        let conv = Conv1dLayer::new(1, num_filters, kernel_size, &mut rng);
        // After global average pooling, output dimension is num_filters
        let dense1 = DenseLayer::new(num_filters, 16, Activation::ReLU, &mut rng);
        let dense_out = DenseLayer::new(16, out_dim, Activation::Identity, &mut rng);

        Self {
            conv,
            dense1,
            dense_out,
        }
    }

    pub fn forward(&self, features: &[f64]) -> Vec<f64> {
        // Reshape 1D vector to [1][L]
        let input = vec![features.to_vec()];
        let conv_out = self.conv.forward(&input);

        // Global Average Pooling: [num_filters][L] -> [num_filters]
        let pooled: Vec<f64> = conv_out
            .iter()
            .map(|f_map| {
                if f_map.is_empty() {
                    0.0
                } else {
                    f_map.iter().sum::<f64>() / f_map.len() as f64
                }
            })
            .collect();

        let (_, d1) = self.dense1.forward(&pooled);
        let (_, out) = self.dense_out.forward(&d1);
        out
    }

    pub fn train_classification(
        &mut self,
        x: &[Vec<f64>],
        y_indices: &[usize],
        _num_classes: usize,
        epochs: usize,
        learning_rate: f64,
    ) {
        let n_samples = x.len();
        if n_samples == 0 {
            return;
        }

        for _ in 0..epochs {
            for (sample_x, &target_class) in x.iter().zip(y_indices.iter()) {
                let input = vec![sample_x.clone()];
                let conv_out = self.conv.forward(&input);

                let pooled: Vec<f64> = conv_out
                    .iter()
                    .map(|f_map| {
                        if f_map.is_empty() {
                            0.0
                        } else {
                            f_map.iter().sum::<f64>() / f_map.len() as f64
                        }
                    })
                    .collect();

                let (d1_pre, d1_post) = self.dense1.forward(&pooled);
                let (_, out_post) = self.dense_out.forward(&d1_post);

                let max_val = out_post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_scores: Vec<f64> = out_post.iter().map(|v| (v - max_val).exp()).collect();
                let sum_exp: f64 = exp_scores.iter().sum::<f64>().max(1e-15);
                let probs: Vec<f64> = exp_scores.iter().map(|v| v / sum_exp).collect();

                let mut out_delta = probs;
                if target_class < out_delta.len() {
                    out_delta[target_class] -= 1.0;
                }

                // Dense out gradients
                let mut d1_delta = vec![0.0; self.dense1.out_features];
                for o in 0..self.dense_out.out_features {
                    let d = out_delta[o];
                    self.dense_out.biases[o] -= learning_rate * d;
                    for h in 0..self.dense_out.in_features {
                        d1_delta[h] += self.dense_out.weights[o][h] * d;
                        self.dense_out.weights[o][h] -= learning_rate * d * d1_post[h];
                    }
                }

                // Dense1 gradients
                for h in 0..self.dense1.out_features {
                    d1_delta[h] *= self.dense1.activation.derivative(d1_pre[h]);
                }
                for h in 0..self.dense1.out_features {
                    let d = d1_delta[h];
                    self.dense1.biases[h] -= learning_rate * d;
                    for p in 0..self.dense1.in_features {
                        self.dense1.weights[h][p] -= learning_rate * d * pooled[p];
                    }
                }
            }
        }
    }

    pub fn train_regression(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        epochs: usize,
        learning_rate: f64,
    ) {
        let n_samples = x.len();
        if n_samples == 0 {
            return;
        }

        for _ in 0..epochs {
            for (sample_x, &target_val) in x.iter().zip(y.iter()) {
                let input = vec![sample_x.clone()];
                let conv_out = self.conv.forward(&input);

                let pooled: Vec<f64> = conv_out
                    .iter()
                    .map(|f_map| {
                        if f_map.is_empty() {
                            0.0
                        } else {
                            f_map.iter().sum::<f64>() / f_map.len() as f64
                        }
                    })
                    .collect();

                let (d1_pre, d1_post) = self.dense1.forward(&pooled);
                let (_, out_post) = self.dense_out.forward(&d1_post);

                let pred = out_post[0];
                let out_delta = vec![pred - target_val];

                let mut d1_delta = vec![0.0; self.dense1.out_features];
                for o in 0..self.dense_out.out_features {
                    let d = out_delta[o];
                    self.dense_out.biases[o] -= learning_rate * d;
                    for h in 0..self.dense_out.in_features {
                        d1_delta[h] += self.dense_out.weights[o][h] * d;
                        self.dense_out.weights[o][h] -= learning_rate * d * d1_post[h];
                    }
                }

                for h in 0..self.dense1.out_features {
                    d1_delta[h] *= self.dense1.activation.derivative(d1_pre[h]);
                }
                for h in 0..self.dense1.out_features {
                    let d = d1_delta[h];
                    self.dense1.biases[h] -= learning_rate * d;
                    for p in 0..self.dense1.in_features {
                        self.dense1.weights[h][p] -= learning_rate * d * pooled[p];
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------
// High-level integration functions
// -------------------------------------------------------------

pub fn run_mlp_classification(
    x_train: &[Vec<f64>],
    y_idx: &[usize],
    classes: &[String],
    x_test: &[Vec<f64>],
) -> Result<Vec<String>, MlError> {
    if x_train.is_empty() {
        return Ok(vec![]);
    }
    let n_features = x_train[0].len();
    let n_classes = classes.len();

    let layer_sizes = vec![n_features, 32, 16, n_classes];
    let activations = vec![Activation::ReLU, Activation::ReLU, Activation::Identity];
    let mut mlp = MlpNetwork::new(&layer_sizes, &activations, 42);

    mlp.train_classification(x_train, y_idx, n_classes, 75, 0.05);

    let predictions = x_test
        .iter()
        .map(|sample| {
            let scores = mlp.forward(sample);
            let best_idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0);
            classes[best_idx].clone()
        })
        .collect();

    Ok(predictions)
}

pub fn run_mlp_regression(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    x_test: &[Vec<f64>],
) -> Result<Vec<f64>, MlError> {
    if x_train.is_empty() {
        return Ok(vec![]);
    }
    let n_features = x_train[0].len();

    let layer_sizes = vec![n_features, 32, 16, 1];
    let activations = vec![Activation::ReLU, Activation::ReLU, Activation::Identity];
    let mut mlp = MlpNetwork::new(&layer_sizes, &activations, 42);

    mlp.train_regression(x_train, y_train, 80, 0.01);

    let predictions = x_test
        .iter()
        .map(|sample| {
            let scores = mlp.forward(sample);
            scores[0]
        })
        .collect();

    Ok(predictions)
}

pub fn run_cnn_classification(
    x_train: &[Vec<f64>],
    y_idx: &[usize],
    classes: &[String],
    x_test: &[Vec<f64>],
) -> Result<Vec<String>, MlError> {
    if x_train.is_empty() {
        return Ok(vec![]);
    }
    let n_features = x_train[0].len();
    let n_classes = classes.len();

    let mut cnn = CnnModel::new(n_features, n_classes, 42);
    cnn.train_classification(x_train, y_idx, n_classes, 80, 0.05);

    let predictions = x_test
        .iter()
        .map(|sample| {
            let scores = cnn.forward(sample);
            let best_idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0);
            classes[best_idx].clone()
        })
        .collect();

    Ok(predictions)
}

pub fn run_cnn_regression(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    x_test: &[Vec<f64>],
) -> Result<Vec<f64>, MlError> {
    if x_train.is_empty() {
        return Ok(vec![]);
    }
    let n_features = x_train[0].len();

    let mut cnn = CnnModel::new(n_features, 1, 42);
    cnn.train_regression(x_train, y_train, 80, 0.01);

    let predictions = x_test
        .iter()
        .map(|sample| {
            let scores = cnn.forward(sample);
            scores[0]
        })
        .collect();

    Ok(predictions)
}
