// Generated from ONNX "model/g_map_se_b4.onnx" by burn-onnx
use burn::prelude::*;
use burn::nn::InstanceNorm;
use burn::nn::InstanceNormConfig;
use burn::nn::LayerNorm;
use burn::nn::LayerNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LinearLayout;
use burn::nn::PRelu;
use burn::nn::PReluConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

fn fast_bigru_forward<B: Backend>(
    gru: &burn::nn::gru::BiGru<B>,
    input: Tensor<B, 3>,
    state: Tensor<B, 3>,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let input = if gru.batch_first {
        input
    } else {
        input.swap_dims(0, 1)
    };

    let [batch_size, _, _] = input.dims();
    let hidden = gru.d_hidden;
    let init_forward = state
        .clone()
        .slice([0..1, 0..batch_size, 0..hidden])
        .squeeze_dim(0);
    let init_reverse = state
        .slice([1..2, 0..batch_size, 0..hidden])
        .squeeze_dim(0);

    let (forward, forward_state) =
        fast_gru_direction(&gru.forward, input.clone(), init_forward, false);
    let (reverse, reverse_state) = fast_gru_direction(&gru.reverse, input, init_reverse, true);
    let output = Tensor::cat([forward, reverse].into(), 2);
    let output = if gru.batch_first {
        output
    } else {
        output.swap_dims(0, 1)
    };
    let final_state = Tensor::cat(
        [
            forward_state.unsqueeze_dim(0),
            reverse_state.unsqueeze_dim(0),
        ]
        .into(),
        0,
    );

    (output, final_state)
}

fn fast_gru_direction<B: Backend>(
    gru: &burn::nn::gru::Gru<B>,
    input: Tensor<B, 3>,
    initial_hidden: Tensor<B, 2>,
    reverse: bool,
) -> (Tensor<B, 3>, Tensor<B, 2>) {
    debug_assert!(
        !gru.reset_after,
        "fast_gru_direction matches reset_after(false) GRUs"
    );

    let [batch_size, seq_len, input_size] = input.dims();
    let hidden = gru.d_hidden;
    let dtype = input.dtype();
    let device = input.device();

    let input_weight = Tensor::cat(
        [
            gru.update_gate.input_transform.weight.val(),
            gru.reset_gate.input_transform.weight.val(),
            gru.new_gate.input_transform.weight.val(),
        ]
        .into(),
        1,
    );
    let input_bias = Tensor::cat(
        [
            gru.update_gate
                .input_transform
                .bias
                .as_ref()
                .expect("generated GRU has input update bias")
                .val(),
            gru.reset_gate
                .input_transform
                .bias
                .as_ref()
                .expect("generated GRU has input reset bias")
                .val(),
            gru.new_gate
                .input_transform
                .bias
                .as_ref()
                .expect("generated GRU has input new bias")
                .val(),
        ]
        .into(),
        0,
    )
    .unsqueeze();
    let input_projected = input
        .reshape([batch_size * seq_len, input_size])
        .matmul(input_weight)
        .reshape([batch_size, seq_len, hidden * 3])
        + input_bias;

    let hidden_zr_weight = Tensor::cat(
        [
            gru.update_gate.hidden_transform.weight.val(),
            gru.reset_gate.hidden_transform.weight.val(),
        ]
        .into(),
        1,
    );
    let hidden_zr_bias = Tensor::cat(
        [
            gru.update_gate
                .hidden_transform
                .bias
                .as_ref()
                .expect("generated GRU has hidden update bias")
                .val(),
            gru.reset_gate
                .hidden_transform
                .bias
                .as_ref()
                .expect("generated GRU has hidden reset bias")
                .val(),
        ]
        .into(),
        0,
    )
    .unsqueeze();
    let hidden_new_weight = gru.new_gate.hidden_transform.weight.val();
    let hidden_new_bias = gru
        .new_gate
        .hidden_transform
        .bias
        .as_ref()
        .expect("generated GRU has hidden new bias")
        .val()
        .unsqueeze();

    let mut hidden_t = initial_hidden;
    let mut outputs = Vec::with_capacity(seq_len);

    for step in 0..seq_len {
        let t = if reverse { seq_len - 1 - step } else { step };
        let input_t = input_projected
            .clone()
            .slice([0..batch_size, t..(t + 1), 0..(hidden * 3)])
            .squeeze_dim(1);
        let input_z = input_t.clone().slice([0..batch_size, 0..hidden]);
        let input_r = input_t
            .clone()
            .slice([0..batch_size, hidden..(hidden * 2)]);
        let input_n = input_t.slice([0..batch_size, (hidden * 2)..(hidden * 3)]);

        let hidden_zr = hidden_t.clone().matmul(hidden_zr_weight.clone()) + hidden_zr_bias.clone();
        let hidden_z = hidden_zr.clone().slice([0..batch_size, 0..hidden]);
        let hidden_r = hidden_zr.slice([0..batch_size, hidden..(hidden * 2)]);

        let update_values = gru.gate_activation.forward(input_z + hidden_z);
        let reset_values = gru.gate_activation.forward(input_r + hidden_r);
        let reset_hidden = hidden_t.clone().mul(reset_values);
        let candidate_state = gru.hidden_activation.forward(
            input_n + reset_hidden.matmul(hidden_new_weight.clone()) + hidden_new_bias.clone(),
        );

        let one_minus_z = update_values.clone().neg().add_scalar(1.0);
        hidden_t = candidate_state.mul(one_minus_z) + update_values.mul(hidden_t);

        if let Some(clip) = gru.clip {
            hidden_t = hidden_t.clamp(-clip, clip);
        }

        outputs.push(hidden_t.clone().unsqueeze_dim(1));
    }

    if reverse {
        outputs.reverse();
    }

    let output = if outputs.is_empty() {
        Tensor::zeros([batch_size, 0, hidden], (&device, dtype))
    } else {
        Tensor::cat(outputs, 1)
    };

    (output, hidden_t)
}


#[derive(Module, Debug)]
pub struct Submodule1<B: Backend> {
    conv2d1: Conv2d<B>,
    instancenormalization1: InstanceNorm<B>,
    linear1: Linear<B>,
    prelu1: PRelu<B>,
    constant10: burn::module::Param<Tensor<B, 1>>,
    conv2d2: Conv2d<B>,
    instancenormalization2: InstanceNorm<B>,
    linear2: Linear<B>,
    prelu2: PRelu<B>,
    linear3: Linear<B>,
    conv2d3: Conv2d<B>,
    instancenormalization3: InstanceNorm<B>,
    prelu3: PRelu<B>,
    conv2d4: Conv2d<B>,
    instancenormalization4: InstanceNorm<B>,
    prelu4: PRelu<B>,
    conv2d5: Conv2d<B>,
    instancenormalization5: InstanceNorm<B>,
    prelu5: PRelu<B>,
    conv2d6: Conv2d<B>,
    instancenormalization6: InstanceNorm<B>,
    prelu6: PRelu<B>,
    linear4: Linear<B>,
    linear5: Linear<B>,
    constant47: burn::module::Param<Tensor<B, 1>>,
    layernormalization1: LayerNorm<B>,
    linear6: Linear<B>,
    constant60: burn::module::Param<Tensor<B, 1>>,
    linear7: Linear<B>,
    layernormalization2: LayerNorm<B>,
    constant70: burn::module::Param<Tensor<B, 3>>,
    gru1: burn::nn::gru::BiGru<B>,
    linear8: Linear<B>,
    layernormalization3: LayerNorm<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule1<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([2, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization1 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let linear1 = LinearConfig::new(192, 192).with_bias(false).init(device);
        let prelu1 = PReluConfig::new().with_num_parameters(64).init(device);
        let constant10: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.20000000298023224f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d2 = Conv2dConfig::new([64, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization2 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let linear2 = LinearConfig::new(192, 192).with_bias(false).init(device);
        let prelu2 = PReluConfig::new().with_num_parameters(64).init(device);
        let linear3 = LinearConfig::new(192, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv2d3 = Conv2dConfig::new([128, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 1, 0, 1))
            .with_dilation([2, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization3 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu3 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d4 = Conv2dConfig::new([192, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 1, 0, 1))
            .with_dilation([4, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization4 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu4 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d5 = Conv2dConfig::new([256, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(8, 1, 0, 1))
            .with_dilation([8, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization5 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu5 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d6 = Conv2dConfig::new([64, 64], [1, 3])
            .with_stride([1, 2])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization6 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu6 = PReluConfig::new().with_num_parameters(64).init(device);
        let linear4 = LinearConfig::new(64, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear5 = LinearConfig::new(128, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let constant47: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data(
                burn::tensor::TensorData::from([1f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let layernormalization1 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear6 = LinearConfig::new(64, 192)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let constant60: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.25f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear7 = LinearConfig::new(64, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization2 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let constant70: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                3,
            >::zeros([2, 321, 128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [2, 321, 128].into(),
        );
        let gru1 = burn::nn::gru::BiGruConfig::new(64, 128, true)
            .with_reset_after(false)
            .with_batch_first(false)
            .init(device);
        let linear8 = LinearConfig::new(256, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization3 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            instancenormalization1,
            linear1,
            prelu1,
            constant10,
            conv2d2,
            instancenormalization2,
            linear2,
            prelu2,
            linear3,
            conv2d3,
            instancenormalization3,
            prelu3,
            conv2d4,
            instancenormalization4,
            prelu4,
            conv2d5,
            instancenormalization5,
            prelu5,
            conv2d6,
            instancenormalization6,
            prelu6,
            linear4,
            linear5,
            constant47,
            layernormalization1,
            linear6,
            constant60,
            linear7,
            layernormalization2,
            constant70,
            gru1,
            linear8,
            layernormalization3,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        noisy_amp: Tensor<B, 3>,
        noisy_pha: Tensor<B, 3>,
        prior_embedding: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let unsqueeze1_out1: Tensor<B, 4> = noisy_amp.unsqueeze_dims::<4>(&[-1]);
        let unsqueeze2_out1: Tensor<B, 4> = noisy_pha.unsqueeze_dims::<4>(&[-1]);
        let reducel21_out1 = {
            prior_embedding
                .clone()
                .square()
                .sum_dim(1usize)
                .sqrt()
        };
        let concat1_out1 = burn::tensor::Tensor::cat(
            [unsqueeze1_out1, unsqueeze2_out1].into(),
            3,
        );
        let clip1_out1 = {
            let __clip_min = 0.0000000000009999999960041972f64;
            reducel21_out1.clamp_min(__clip_min)
        };
        let transpose1_out1 = concat1_out1.permute([0, 3, 2, 1]);
        let expand1_out1 = {
            let onnx_shape: [i64; 2usize] = [4, 192];
            let input_dims = clip1_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..2usize {
                let dim_offset = 2usize - 2usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            clip1_out1.expand(shape)
        };
        let conv2d1_out1 = self.conv2d1.forward(transpose1_out1);
        let div1_out1 = prior_embedding.div(expand1_out1);
        let instancenormalization1_out1 = self
            .instancenormalization1
            .forward(conv2d1_out1);
        let linear1_out1 = self.linear1.forward(div1_out1);
        let prelu1_out1 = self.prelu1.forward(instancenormalization1_out1);
        let constant10_out1 = self.constant10.val();
        let div2_out1 = linear1_out1.div((constant10_out1).unsqueeze_dims(&[0isize]));
        let conv2d2_out1 = self.conv2d2.forward(prelu1_out1.clone());
        let softmax1_out1 = burn::tensor::activation::softmax(div2_out1, 1);
        let instancenormalization2_out1 = self
            .instancenormalization2
            .forward(conv2d2_out1);
        let linear2_out1 = self.linear2.forward(softmax1_out1);
        let prelu2_out1 = self.prelu2.forward(instancenormalization2_out1);
        let linear3_out1 = self.linear3.forward(linear2_out1);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [prelu2_out1, prelu1_out1].into(),
            1,
        );
        let relu1_out1 = burn::tensor::activation::relu(linear3_out1);
        let conv2d3_out1 = self.conv2d3.forward(concat2_out1.clone());
        let unsqueeze3_out1: Tensor<B, 4> = relu1_out1.unsqueeze_dims::<4>(&[1, 2]);
        let instancenormalization3_out1 = self
            .instancenormalization3
            .forward(conv2d3_out1);
        let expand2_out1 = {
            let onnx_shape: [i64; 4usize] = [4, 321, 101, 64];
            let input_dims = unsqueeze3_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze3_out1.expand(shape)
        };
        let prelu3_out1 = self.prelu3.forward(instancenormalization3_out1);
        let concat3_out1 = burn::tensor::Tensor::cat(
            [prelu3_out1, concat2_out1].into(),
            1,
        );
        let conv2d4_out1 = self.conv2d4.forward(concat3_out1.clone());
        let instancenormalization4_out1 = self
            .instancenormalization4
            .forward(conv2d4_out1);
        let prelu4_out1 = self.prelu4.forward(instancenormalization4_out1);
        let concat4_out1 = burn::tensor::Tensor::cat(
            [prelu4_out1, concat3_out1].into(),
            1,
        );
        let conv2d5_out1 = self.conv2d5.forward(concat4_out1);
        let instancenormalization5_out1 = self
            .instancenormalization5
            .forward(conv2d5_out1);
        let prelu5_out1 = self.prelu5.forward(instancenormalization5_out1);
        let conv2d6_out1 = self.conv2d6.forward(prelu5_out1);
        let instancenormalization6_out1 = self
            .instancenormalization6
            .forward(conv2d6_out1);
        let prelu6_out1 = self.prelu6.forward(instancenormalization6_out1);
        let transpose2_out1 = prelu6_out1.permute([0, 2, 3, 1]);
        let reshape1_out1 = transpose2_out1.reshape([-1, 64]);
        let linear4_out1 = self.linear4.forward(reshape1_out1);
        let reshape2_out1 = linear4_out1.reshape([4, 321, 101, 64]);
        let relu2_out1 = burn::tensor::activation::relu(reshape2_out1);
        let concat5_out1 = burn::tensor::Tensor::cat(
            [relu2_out1.clone(), expand2_out1.clone()].into(),
            3,
        );
        let reshape3_out1 = concat5_out1.reshape([-1, 128]);
        let linear5_out1 = self.linear5.forward(reshape3_out1);
        let reshape4_out1 = linear5_out1.reshape([4, 321, 101, 64]);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(reshape4_out1);
        let constant47_out1 = self.constant47.val();
        let sub1_out1 = (constant47_out1)
            .unsqueeze_dims(&[0isize, 1isize, 2isize])
            .sub(sigmoid1_out1.clone());
        let mul1_out1 = sigmoid1_out1.mul(expand2_out1);
        let mul2_out1 = sub1_out1.mul(relu2_out1);
        let add1_out1 = mul2_out1.add(mul1_out1);
        let transpose3_out1 = add1_out1.permute([0, 2, 1, 3]);
        let reshape5_out1 = transpose3_out1.reshape([404, 321, 64]);
        let layernormalization1_out1 = {
            self.layernormalization1
                .forward(reshape5_out1.clone())
        };
        let transpose4_out1 = layernormalization1_out1.permute([1, 0, 2]);
        let reshape6_out1 = transpose4_out1.reshape([-1, 64]);
        let linear6_out1 = self.linear6.forward(reshape6_out1);
        let reshape7_out1 = linear6_out1.reshape([321, 404, 3, 64]);
        let unsqueeze4_out1: Tensor<B, 5> = reshape7_out1.unsqueeze_dims::<5>(&[0]);
        let transpose5_out1 = unsqueeze4_out1.permute([3, 1, 2, 0, 4]);
        let squeeze1_out1 = transpose5_out1.squeeze_dims::<4>(&[-2]);
        let gather1_out1 = {
            let sliced = squeeze1_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather2_out1 = {
            let sliced = squeeze1_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather3_out1 = {
            let sliced = squeeze1_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape8_out1 = gather1_out1.reshape([321, 1616, 16]);
        let reshape9_out1 = gather2_out1.reshape([321, 1616, 16]);
        let reshape10_out1 = gather3_out1.reshape([321, 1616, 16]);
        let transpose6_out1 = reshape8_out1.permute([1, 0, 2]);
        let transpose7_out1 = reshape10_out1.permute([1, 0, 2]);
        let transpose8_out1 = reshape9_out1.permute([1, 2, 0]);
        let constant60_out1 = self.constant60.val();
        let mul3_out1 = transpose6_out1
            .mul((constant60_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let matmul3_out1 = mul3_out1.matmul(transpose8_out1);
        let softmax2_out1 = burn::tensor::activation::softmax(matmul3_out1, 2);
        let matmul4_out1 = softmax2_out1.matmul(transpose7_out1);
        let transpose9_out1 = matmul4_out1.permute([1, 0, 2]);
        let reshape11_out1 = transpose9_out1.reshape([129684, 64]);
        let linear7_out1 = self.linear7.forward(reshape11_out1);
        let reshape12_out1 = linear7_out1.reshape([321, 404, 64]);
        let transpose10_out1 = reshape12_out1.permute([1, 0, 2]);
        let add2_out1 = reshape5_out1.clone().add(transpose10_out1);
        let layernormalization2_out1 = {
            self.layernormalization2
                .forward(add2_out1.clone())
        };
        let constant70_out1 = self.constant70.val();
        let gru1_out1 = {
            let (output_seq, _final_state) =
                fast_bigru_forward(&self.gru1, layernormalization2_out1, constant70_out1);
            {
                let [seq_len, batch_size, _] = output_seq.dims();
                let reshaped = output_seq.reshape([seq_len, batch_size, 2, 128usize]);
                reshaped.swap_dims(1, 2)
            }
        };
        let transpose11_out1 = gru1_out1.permute([0, 2, 1, 3]);
        let reshape13_out1 = transpose11_out1.reshape([404, 321, 256]);
        let leakyrelu1_out1 = burn::tensor::activation::leaky_relu(
            reshape13_out1,
            0.009999999776482582,
        );
        let reshape14_out1 = leakyrelu1_out1.reshape([-1, 256]);
        let linear8_out1 = self.linear8.forward(reshape14_out1);
        let reshape15_out1 = linear8_out1.reshape([404, 321, 64]);
        let add3_out1 = add2_out1.add(reshape15_out1);
        let layernormalization3_out1 = {
            self.layernormalization3
                .forward(add3_out1)
        };
        let add4_out1 = layernormalization3_out1.add(reshape5_out1);
        (add4_out1, constant60_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    layernormalization4: LayerNorm<B>,
    linear9: Linear<B>,
    linear10: Linear<B>,
    layernormalization5: LayerNorm<B>,
    constant93: burn::module::Param<Tensor<B, 3>>,
    gru2: burn::nn::gru::BiGru<B>,
    linear11: Linear<B>,
    layernormalization6: LayerNorm<B>,
    layernormalization7: LayerNorm<B>,
    linear12: Linear<B>,
    linear13: Linear<B>,
    layernormalization8: LayerNorm<B>,
    constant110: burn::module::Param<Tensor<B, 3>>,
    gru3: burn::nn::gru::BiGru<B>,
    linear14: Linear<B>,
    layernormalization9: LayerNorm<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization4 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear9 = LinearConfig::new(64, 192)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear10 = LinearConfig::new(64, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization5 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let constant93: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                3,
            >::zeros([2, 101, 128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [2, 101, 128].into(),
        );
        let gru2 = burn::nn::gru::BiGruConfig::new(64, 128, true)
            .with_reset_after(false)
            .with_batch_first(false)
            .init(device);
        let linear11 = LinearConfig::new(256, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization6 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let layernormalization7 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear12 = LinearConfig::new(64, 192)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear13 = LinearConfig::new(64, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization8 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let constant110: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                3,
            >::zeros([2, 321, 128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [2, 321, 128].into(),
        );
        let gru3 = burn::nn::gru::BiGruConfig::new(64, 128, true)
            .with_reset_after(false)
            .with_batch_first(false)
            .init(device);
        let linear14 = LinearConfig::new(256, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization9 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        Self {
            layernormalization4,
            linear9,
            linear10,
            layernormalization5,
            constant93,
            gru2,
            linear11,
            layernormalization6,
            layernormalization7,
            linear12,
            linear13,
            layernormalization8,
            constant110,
            gru3,
            linear14,
            layernormalization9,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add4_out1: Tensor<B, 3>,
        constant60_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reshape16_out1 = add4_out1.reshape([4, 101, 321, 64]);
        let transpose12_out1 = reshape16_out1.permute([0, 2, 1, 3]);
        let reshape17_out1 = transpose12_out1.reshape([1284, 101, 64]);
        let layernormalization4_out1 = {
            self.layernormalization4
                .forward(reshape17_out1.clone())
        };
        let transpose13_out1 = layernormalization4_out1.permute([1, 0, 2]);
        let reshape18_out1 = transpose13_out1.reshape([-1, 64]);
        let linear9_out1 = self.linear9.forward(reshape18_out1);
        let reshape19_out1 = linear9_out1.reshape([101, 1284, 3, 64]);
        let unsqueeze5_out1: Tensor<B, 5> = reshape19_out1.unsqueeze_dims::<5>(&[0]);
        let transpose14_out1 = unsqueeze5_out1.permute([3, 1, 2, 0, 4]);
        let squeeze2_out1 = transpose14_out1.squeeze_dims::<4>(&[-2]);
        let gather4_out1 = {
            let sliced = squeeze2_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather5_out1 = {
            let sliced = squeeze2_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather6_out1 = {
            let sliced = squeeze2_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape20_out1 = gather4_out1.reshape([101, 5136, 16]);
        let reshape21_out1 = gather5_out1.reshape([101, 5136, 16]);
        let reshape22_out1 = gather6_out1.reshape([101, 5136, 16]);
        let transpose15_out1 = reshape20_out1.permute([1, 0, 2]);
        let transpose16_out1 = reshape22_out1.permute([1, 0, 2]);
        let transpose17_out1 = reshape21_out1.permute([1, 2, 0]);
        let mul4_out1 = transpose15_out1
            .mul((constant60_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let matmul5_out1 = mul4_out1.matmul(transpose17_out1);
        let softmax3_out1 = burn::tensor::activation::softmax(matmul5_out1, 2);
        let matmul6_out1 = softmax3_out1.matmul(transpose16_out1);
        let transpose18_out1 = matmul6_out1.permute([1, 0, 2]);
        let reshape23_out1 = transpose18_out1.reshape([129684, 64]);
        let linear10_out1 = self.linear10.forward(reshape23_out1);
        let reshape24_out1 = linear10_out1.reshape([101, 1284, 64]);
        let transpose19_out1 = reshape24_out1.permute([1, 0, 2]);
        let add5_out1 = reshape17_out1.clone().add(transpose19_out1);
        let layernormalization5_out1 = {
            self.layernormalization5
                .forward(add5_out1.clone())
        };
        let constant93_out1 = self.constant93.val();
        let gru2_out1 = {
            let (output_seq, _final_state) =
                fast_bigru_forward(&self.gru2, layernormalization5_out1, constant93_out1);
            {
                let [seq_len, batch_size, _] = output_seq.dims();
                let reshaped = output_seq.reshape([seq_len, batch_size, 2, 128usize]);
                reshaped.swap_dims(1, 2)
            }
        };
        let transpose20_out1 = gru2_out1.permute([0, 2, 1, 3]);
        let reshape25_out1 = transpose20_out1.reshape([1284, 101, 256]);
        let leakyrelu2_out1 = burn::tensor::activation::leaky_relu(
            reshape25_out1,
            0.009999999776482582,
        );
        let reshape26_out1 = leakyrelu2_out1.reshape([-1, 256]);
        let linear11_out1 = self.linear11.forward(reshape26_out1);
        let reshape27_out1 = linear11_out1.reshape([1284, 101, 64]);
        let add6_out1 = add5_out1.add(reshape27_out1);
        let layernormalization6_out1 = {
            self.layernormalization6
                .forward(add6_out1)
        };
        let add7_out1 = layernormalization6_out1.add(reshape17_out1);
        let reshape28_out1 = add7_out1.reshape([4, 321, 101, 64]);
        let transpose21_out1 = reshape28_out1.permute([0, 2, 1, 3]);
        let reshape29_out1 = transpose21_out1.reshape([404, 321, 64]);
        let layernormalization7_out1 = {
            self.layernormalization7
                .forward(reshape29_out1.clone())
        };
        let transpose22_out1 = layernormalization7_out1.permute([1, 0, 2]);
        let reshape30_out1 = transpose22_out1.reshape([-1, 64]);
        let linear12_out1 = self.linear12.forward(reshape30_out1);
        let reshape31_out1 = linear12_out1.reshape([321, 404, 3, 64]);
        let unsqueeze6_out1: Tensor<B, 5> = reshape31_out1.unsqueeze_dims::<5>(&[0]);
        let transpose23_out1 = unsqueeze6_out1.permute([3, 1, 2, 0, 4]);
        let squeeze3_out1 = transpose23_out1.squeeze_dims::<4>(&[-2]);
        let gather7_out1 = {
            let sliced = squeeze3_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather8_out1 = {
            let sliced = squeeze3_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather9_out1 = {
            let sliced = squeeze3_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape32_out1 = gather7_out1.reshape([321, 1616, 16]);
        let reshape33_out1 = gather8_out1.reshape([321, 1616, 16]);
        let reshape34_out1 = gather9_out1.reshape([321, 1616, 16]);
        let transpose24_out1 = reshape32_out1.permute([1, 0, 2]);
        let transpose25_out1 = reshape34_out1.permute([1, 0, 2]);
        let transpose26_out1 = reshape33_out1.permute([1, 2, 0]);
        let mul5_out1 = transpose24_out1
            .mul((constant60_out1).unsqueeze_dims(&[0isize, 1isize]));
        let matmul7_out1 = mul5_out1.matmul(transpose26_out1);
        let softmax4_out1 = burn::tensor::activation::softmax(matmul7_out1, 2);
        let matmul8_out1 = softmax4_out1.matmul(transpose25_out1);
        let transpose27_out1 = matmul8_out1.permute([1, 0, 2]);
        let reshape35_out1 = transpose27_out1.reshape([129684, 64]);
        let linear13_out1 = self.linear13.forward(reshape35_out1);
        let reshape36_out1 = linear13_out1.reshape([321, 404, 64]);
        let transpose28_out1 = reshape36_out1.permute([1, 0, 2]);
        let add8_out1 = reshape29_out1.clone().add(transpose28_out1);
        let layernormalization8_out1 = {
            self.layernormalization8
                .forward(add8_out1.clone())
        };
        let constant110_out1 = self.constant110.val();
        let gru3_out1 = {
            let (output_seq, _final_state) =
                fast_bigru_forward(&self.gru3, layernormalization8_out1, constant110_out1);
            {
                let [seq_len, batch_size, _] = output_seq.dims();
                let reshaped = output_seq.reshape([seq_len, batch_size, 2, 128usize]);
                reshaped.swap_dims(1, 2)
            }
        };
        let transpose29_out1 = gru3_out1.permute([0, 2, 1, 3]);
        let reshape37_out1 = transpose29_out1.reshape([404, 321, 256]);
        let leakyrelu3_out1 = burn::tensor::activation::leaky_relu(
            reshape37_out1,
            0.009999999776482582,
        );
        let reshape38_out1 = leakyrelu3_out1.reshape([-1, 256]);
        let linear14_out1 = self.linear14.forward(reshape38_out1);
        let reshape39_out1 = linear14_out1.reshape([404, 321, 64]);
        let add9_out1 = add8_out1.add(reshape39_out1);
        let layernormalization9_out1 = {
            self.layernormalization9
                .forward(add9_out1)
        };
        let add10_out1 = layernormalization9_out1.add(reshape29_out1);
        add10_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    layernormalization10: LayerNorm<B>,
    linear15: Linear<B>,
    linear16: Linear<B>,
    layernormalization11: LayerNorm<B>,
    constant126: burn::module::Param<Tensor<B, 3>>,
    gru4: burn::nn::gru::BiGru<B>,
    linear17: Linear<B>,
    layernormalization12: LayerNorm<B>,
    layernormalization13: LayerNorm<B>,
    linear18: Linear<B>,
    linear19: Linear<B>,
    layernormalization14: LayerNorm<B>,
    constant142: burn::module::Param<Tensor<B, 3>>,
    gru5: burn::nn::gru::BiGru<B>,
    linear20: Linear<B>,
    layernormalization15: LayerNorm<B>,
    layernormalization16: LayerNorm<B>,
    linear21: Linear<B>,
    linear22: Linear<B>,
    layernormalization17: LayerNorm<B>,
    constant158: burn::module::Param<Tensor<B, 3>>,
    gru6: burn::nn::gru::BiGru<B>,
    linear23: Linear<B>,
    layernormalization18: LayerNorm<B>,
    layernormalization19: LayerNorm<B>,
    linear24: Linear<B>,
    linear25: Linear<B>,
    layernormalization20: LayerNorm<B>,
    constant174: burn::module::Param<Tensor<B, 3>>,
    gru7: burn::nn::gru::BiGru<B>,
    linear26: Linear<B>,
    layernormalization21: LayerNorm<B>,
    layernormalization22: LayerNorm<B>,
    linear27: Linear<B>,
    linear28: Linear<B>,
    layernormalization23: LayerNorm<B>,
    constant190: burn::module::Param<Tensor<B, 3>>,
    gru8: burn::nn::gru::BiGru<B>,
    linear29: Linear<B>,
    layernormalization24: LayerNorm<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let layernormalization10 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear15 = LinearConfig::new(64, 192)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear16 = LinearConfig::new(64, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization11 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let constant126: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                3,
            >::zeros([2, 101, 128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [2, 101, 128].into(),
        );
        let gru4 = burn::nn::gru::BiGruConfig::new(64, 128, true)
            .with_reset_after(false)
            .with_batch_first(false)
            .init(device);
        let linear17 = LinearConfig::new(256, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization12 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let layernormalization13 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear18 = LinearConfig::new(64, 192)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear19 = LinearConfig::new(64, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization14 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let constant142: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                3,
            >::zeros([2, 321, 128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [2, 321, 128].into(),
        );
        let gru5 = burn::nn::gru::BiGruConfig::new(64, 128, true)
            .with_reset_after(false)
            .with_batch_first(false)
            .init(device);
        let linear20 = LinearConfig::new(256, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization15 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let layernormalization16 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear21 = LinearConfig::new(64, 192)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear22 = LinearConfig::new(64, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization17 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let constant158: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                3,
            >::zeros([2, 101, 128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [2, 101, 128].into(),
        );
        let gru6 = burn::nn::gru::BiGruConfig::new(64, 128, true)
            .with_reset_after(false)
            .with_batch_first(false)
            .init(device);
        let linear23 = LinearConfig::new(256, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization18 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let layernormalization19 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear24 = LinearConfig::new(64, 192)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear25 = LinearConfig::new(64, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization20 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let constant174: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                3,
            >::zeros([2, 321, 128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [2, 321, 128].into(),
        );
        let gru7 = burn::nn::gru::BiGruConfig::new(64, 128, true)
            .with_reset_after(false)
            .with_batch_first(false)
            .init(device);
        let linear26 = LinearConfig::new(256, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization21 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let layernormalization22 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let linear27 = LinearConfig::new(64, 192)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear28 = LinearConfig::new(64, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization23 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        let constant190: burn::module::Param<Tensor<B, 3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                3,
            >::zeros([2, 101, 128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [2, 101, 128].into(),
        );
        let gru8 = burn::nn::gru::BiGruConfig::new(64, 128, true)
            .with_reset_after(false)
            .with_batch_first(false)
            .init(device);
        let linear29 = LinearConfig::new(256, 64)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let layernormalization24 = LayerNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_bias(true)
            .init(device);
        Self {
            layernormalization10,
            linear15,
            linear16,
            layernormalization11,
            constant126,
            gru4,
            linear17,
            layernormalization12,
            layernormalization13,
            linear18,
            linear19,
            layernormalization14,
            constant142,
            gru5,
            linear20,
            layernormalization15,
            layernormalization16,
            linear21,
            linear22,
            layernormalization17,
            constant158,
            gru6,
            linear23,
            layernormalization18,
            layernormalization19,
            linear24,
            linear25,
            layernormalization20,
            constant174,
            gru7,
            linear26,
            layernormalization21,
            layernormalization22,
            linear27,
            linear28,
            layernormalization23,
            constant190,
            gru8,
            linear29,
            layernormalization24,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add10_out1: Tensor<B, 3>,
        constant60_out1: Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        let reshape40_out1 = add10_out1.reshape([4, 101, 321, 64]);
        let transpose30_out1 = reshape40_out1.permute([0, 2, 1, 3]);
        let reshape41_out1 = transpose30_out1.reshape([1284, 101, 64]);
        let layernormalization10_out1 = {
            self.layernormalization10
                .forward(reshape41_out1.clone())
        };
        let transpose31_out1 = layernormalization10_out1.permute([1, 0, 2]);
        let reshape42_out1 = transpose31_out1.reshape([-1, 64]);
        let linear15_out1 = self.linear15.forward(reshape42_out1);
        let reshape43_out1 = linear15_out1.reshape([101, 1284, 3, 64]);
        let unsqueeze7_out1: Tensor<B, 5> = reshape43_out1.unsqueeze_dims::<5>(&[0]);
        let transpose32_out1 = unsqueeze7_out1.permute([3, 1, 2, 0, 4]);
        let squeeze4_out1 = transpose32_out1.squeeze_dims::<4>(&[-2]);
        let gather10_out1 = {
            let sliced = squeeze4_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather11_out1 = {
            let sliced = squeeze4_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather12_out1 = {
            let sliced = squeeze4_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape44_out1 = gather10_out1.reshape([101, 5136, 16]);
        let reshape45_out1 = gather11_out1.reshape([101, 5136, 16]);
        let reshape46_out1 = gather12_out1.reshape([101, 5136, 16]);
        let transpose33_out1 = reshape44_out1.permute([1, 0, 2]);
        let transpose34_out1 = reshape46_out1.permute([1, 0, 2]);
        let transpose35_out1 = reshape45_out1.permute([1, 2, 0]);
        let mul6_out1 = transpose33_out1
            .mul((constant60_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let matmul9_out1 = mul6_out1.matmul(transpose35_out1);
        let softmax5_out1 = burn::tensor::activation::softmax(matmul9_out1, 2);
        let matmul10_out1 = softmax5_out1.matmul(transpose34_out1);
        let transpose36_out1 = matmul10_out1.permute([1, 0, 2]);
        let reshape47_out1 = transpose36_out1.reshape([129684, 64]);
        let linear16_out1 = self.linear16.forward(reshape47_out1);
        let reshape48_out1 = linear16_out1.reshape([101, 1284, 64]);
        let transpose37_out1 = reshape48_out1.permute([1, 0, 2]);
        let add11_out1 = reshape41_out1.clone().add(transpose37_out1);
        let layernormalization11_out1 = {
            self.layernormalization11
                .forward(add11_out1.clone())
        };
        let constant126_out1 = self.constant126.val();
        let gru4_out1 = {
            let (output_seq, _final_state) =
                fast_bigru_forward(&self.gru4, layernormalization11_out1, constant126_out1);
            {
                let [seq_len, batch_size, _] = output_seq.dims();
                let reshaped = output_seq.reshape([seq_len, batch_size, 2, 128usize]);
                reshaped.swap_dims(1, 2)
            }
        };
        let transpose38_out1 = gru4_out1.permute([0, 2, 1, 3]);
        let reshape49_out1 = transpose38_out1.reshape([1284, 101, 256]);
        let leakyrelu4_out1 = burn::tensor::activation::leaky_relu(
            reshape49_out1,
            0.009999999776482582,
        );
        let reshape50_out1 = leakyrelu4_out1.reshape([-1, 256]);
        let linear17_out1 = self.linear17.forward(reshape50_out1);
        let reshape51_out1 = linear17_out1.reshape([1284, 101, 64]);
        let add12_out1 = add11_out1.add(reshape51_out1);
        let layernormalization12_out1 = {
            self.layernormalization12
                .forward(add12_out1)
        };
        let add13_out1 = layernormalization12_out1.add(reshape41_out1);
        let reshape52_out1 = add13_out1.reshape([4, 321, 101, 64]);
        let transpose39_out1 = reshape52_out1.permute([0, 2, 1, 3]);
        let reshape53_out1 = transpose39_out1.reshape([404, 321, 64]);
        let layernormalization13_out1 = {
            self.layernormalization13
                .forward(reshape53_out1.clone())
        };
        let transpose40_out1 = layernormalization13_out1.permute([1, 0, 2]);
        let reshape54_out1 = transpose40_out1.reshape([-1, 64]);
        let linear18_out1 = self.linear18.forward(reshape54_out1);
        let reshape55_out1 = linear18_out1.reshape([321, 404, 3, 64]);
        let unsqueeze8_out1: Tensor<B, 5> = reshape55_out1.unsqueeze_dims::<5>(&[0]);
        let transpose41_out1 = unsqueeze8_out1.permute([3, 1, 2, 0, 4]);
        let squeeze5_out1 = transpose41_out1.squeeze_dims::<4>(&[-2]);
        let gather13_out1 = {
            let sliced = squeeze5_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather14_out1 = {
            let sliced = squeeze5_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather15_out1 = {
            let sliced = squeeze5_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape56_out1 = gather13_out1.reshape([321, 1616, 16]);
        let reshape57_out1 = gather14_out1.reshape([321, 1616, 16]);
        let reshape58_out1 = gather15_out1.reshape([321, 1616, 16]);
        let transpose42_out1 = reshape56_out1.permute([1, 0, 2]);
        let transpose43_out1 = reshape58_out1.permute([1, 0, 2]);
        let transpose44_out1 = reshape57_out1.permute([1, 2, 0]);
        let mul7_out1 = transpose42_out1
            .mul((constant60_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let matmul11_out1 = mul7_out1.matmul(transpose44_out1);
        let softmax6_out1 = burn::tensor::activation::softmax(matmul11_out1, 2);
        let matmul12_out1 = softmax6_out1.matmul(transpose43_out1);
        let transpose45_out1 = matmul12_out1.permute([1, 0, 2]);
        let reshape59_out1 = transpose45_out1.reshape([129684, 64]);
        let linear19_out1 = self.linear19.forward(reshape59_out1);
        let reshape60_out1 = linear19_out1.reshape([321, 404, 64]);
        let transpose46_out1 = reshape60_out1.permute([1, 0, 2]);
        let add14_out1 = reshape53_out1.clone().add(transpose46_out1);
        let layernormalization14_out1 = {
            self.layernormalization14
                .forward(add14_out1.clone())
        };
        let constant142_out1 = self.constant142.val();
        let gru5_out1 = {
            let (output_seq, _final_state) =
                fast_bigru_forward(&self.gru5, layernormalization14_out1, constant142_out1);
            {
                let [seq_len, batch_size, _] = output_seq.dims();
                let reshaped = output_seq.reshape([seq_len, batch_size, 2, 128usize]);
                reshaped.swap_dims(1, 2)
            }
        };
        let transpose47_out1 = gru5_out1.permute([0, 2, 1, 3]);
        let reshape61_out1 = transpose47_out1.reshape([404, 321, 256]);
        let leakyrelu5_out1 = burn::tensor::activation::leaky_relu(
            reshape61_out1,
            0.009999999776482582,
        );
        let reshape62_out1 = leakyrelu5_out1.reshape([-1, 256]);
        let linear20_out1 = self.linear20.forward(reshape62_out1);
        let reshape63_out1 = linear20_out1.reshape([404, 321, 64]);
        let add15_out1 = add14_out1.add(reshape63_out1);
        let layernormalization15_out1 = {
            self.layernormalization15
                .forward(add15_out1)
        };
        let add16_out1 = layernormalization15_out1.add(reshape53_out1);
        let reshape64_out1 = add16_out1.reshape([4, 101, 321, 64]);
        let transpose48_out1 = reshape64_out1.permute([0, 2, 1, 3]);
        let reshape65_out1 = transpose48_out1.reshape([1284, 101, 64]);
        let layernormalization16_out1 = {
            self.layernormalization16
                .forward(reshape65_out1.clone())
        };
        let transpose49_out1 = layernormalization16_out1.permute([1, 0, 2]);
        let reshape66_out1 = transpose49_out1.reshape([-1, 64]);
        let linear21_out1 = self.linear21.forward(reshape66_out1);
        let reshape67_out1 = linear21_out1.reshape([101, 1284, 3, 64]);
        let unsqueeze9_out1: Tensor<B, 5> = reshape67_out1.unsqueeze_dims::<5>(&[0]);
        let transpose50_out1 = unsqueeze9_out1.permute([3, 1, 2, 0, 4]);
        let squeeze6_out1 = transpose50_out1.squeeze_dims::<4>(&[-2]);
        let gather16_out1 = {
            let sliced = squeeze6_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather17_out1 = {
            let sliced = squeeze6_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather18_out1 = {
            let sliced = squeeze6_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape68_out1 = gather16_out1.reshape([101, 5136, 16]);
        let reshape69_out1 = gather17_out1.reshape([101, 5136, 16]);
        let reshape70_out1 = gather18_out1.reshape([101, 5136, 16]);
        let transpose51_out1 = reshape68_out1.permute([1, 0, 2]);
        let transpose52_out1 = reshape70_out1.permute([1, 0, 2]);
        let transpose53_out1 = reshape69_out1.permute([1, 2, 0]);
        let mul8_out1 = transpose51_out1
            .mul((constant60_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let matmul13_out1 = mul8_out1.matmul(transpose53_out1);
        let softmax7_out1 = burn::tensor::activation::softmax(matmul13_out1, 2);
        let matmul14_out1 = softmax7_out1.matmul(transpose52_out1);
        let transpose54_out1 = matmul14_out1.permute([1, 0, 2]);
        let reshape71_out1 = transpose54_out1.reshape([129684, 64]);
        let linear22_out1 = self.linear22.forward(reshape71_out1);
        let reshape72_out1 = linear22_out1.reshape([101, 1284, 64]);
        let transpose55_out1 = reshape72_out1.permute([1, 0, 2]);
        let add17_out1 = reshape65_out1.clone().add(transpose55_out1);
        let layernormalization17_out1 = {
            self.layernormalization17
                .forward(add17_out1.clone())
        };
        let constant158_out1 = self.constant158.val();
        let gru6_out1 = {
            let (output_seq, _final_state) =
                fast_bigru_forward(&self.gru6, layernormalization17_out1, constant158_out1);
            {
                let [seq_len, batch_size, _] = output_seq.dims();
                let reshaped = output_seq.reshape([seq_len, batch_size, 2, 128usize]);
                reshaped.swap_dims(1, 2)
            }
        };
        let transpose56_out1 = gru6_out1.permute([0, 2, 1, 3]);
        let reshape73_out1 = transpose56_out1.reshape([1284, 101, 256]);
        let leakyrelu6_out1 = burn::tensor::activation::leaky_relu(
            reshape73_out1,
            0.009999999776482582,
        );
        let reshape74_out1 = leakyrelu6_out1.reshape([-1, 256]);
        let linear23_out1 = self.linear23.forward(reshape74_out1);
        let reshape75_out1 = linear23_out1.reshape([1284, 101, 64]);
        let add18_out1 = add17_out1.add(reshape75_out1);
        let layernormalization18_out1 = {
            self.layernormalization18
                .forward(add18_out1)
        };
        let add19_out1 = layernormalization18_out1.add(reshape65_out1);
        let reshape76_out1 = add19_out1.reshape([4, 321, 101, 64]);
        let transpose57_out1 = reshape76_out1.permute([0, 2, 1, 3]);
        let reshape77_out1 = transpose57_out1.reshape([404, 321, 64]);
        let layernormalization19_out1 = {
            self.layernormalization19
                .forward(reshape77_out1.clone())
        };
        let transpose58_out1 = layernormalization19_out1.permute([1, 0, 2]);
        let reshape78_out1 = transpose58_out1.reshape([-1, 64]);
        let linear24_out1 = self.linear24.forward(reshape78_out1);
        let reshape79_out1 = linear24_out1.reshape([321, 404, 3, 64]);
        let unsqueeze10_out1: Tensor<B, 5> = reshape79_out1.unsqueeze_dims::<5>(&[0]);
        let transpose59_out1 = unsqueeze10_out1.permute([3, 1, 2, 0, 4]);
        let squeeze7_out1 = transpose59_out1.squeeze_dims::<4>(&[-2]);
        let gather19_out1 = {
            let sliced = squeeze7_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather20_out1 = {
            let sliced = squeeze7_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather21_out1 = {
            let sliced = squeeze7_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape80_out1 = gather19_out1.reshape([321, 1616, 16]);
        let reshape81_out1 = gather20_out1.reshape([321, 1616, 16]);
        let reshape82_out1 = gather21_out1.reshape([321, 1616, 16]);
        let transpose60_out1 = reshape80_out1.permute([1, 0, 2]);
        let transpose61_out1 = reshape82_out1.permute([1, 0, 2]);
        let transpose62_out1 = reshape81_out1.permute([1, 2, 0]);
        let mul9_out1 = transpose60_out1
            .mul((constant60_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let matmul15_out1 = mul9_out1.matmul(transpose62_out1);
        let softmax8_out1 = burn::tensor::activation::softmax(matmul15_out1, 2);
        let matmul16_out1 = softmax8_out1.matmul(transpose61_out1);
        let transpose63_out1 = matmul16_out1.permute([1, 0, 2]);
        let reshape83_out1 = transpose63_out1.reshape([129684, 64]);
        let linear25_out1 = self.linear25.forward(reshape83_out1);
        let reshape84_out1 = linear25_out1.reshape([321, 404, 64]);
        let transpose64_out1 = reshape84_out1.permute([1, 0, 2]);
        let add20_out1 = reshape77_out1.clone().add(transpose64_out1);
        let layernormalization20_out1 = {
            self.layernormalization20
                .forward(add20_out1.clone())
        };
        let constant174_out1 = self.constant174.val();
        let gru7_out1 = {
            let (output_seq, _final_state) =
                fast_bigru_forward(&self.gru7, layernormalization20_out1, constant174_out1);
            {
                let [seq_len, batch_size, _] = output_seq.dims();
                let reshaped = output_seq.reshape([seq_len, batch_size, 2, 128usize]);
                reshaped.swap_dims(1, 2)
            }
        };
        let transpose65_out1 = gru7_out1.permute([0, 2, 1, 3]);
        let reshape85_out1 = transpose65_out1.reshape([404, 321, 256]);
        let leakyrelu7_out1 = burn::tensor::activation::leaky_relu(
            reshape85_out1,
            0.009999999776482582,
        );
        let reshape86_out1 = leakyrelu7_out1.reshape([-1, 256]);
        let linear26_out1 = self.linear26.forward(reshape86_out1);
        let reshape87_out1 = linear26_out1.reshape([404, 321, 64]);
        let add21_out1 = add20_out1.add(reshape87_out1);
        let layernormalization21_out1 = {
            self.layernormalization21
                .forward(add21_out1)
        };
        let add22_out1 = layernormalization21_out1.add(reshape77_out1);
        let reshape88_out1 = add22_out1.reshape([4, 101, 321, 64]);
        let transpose66_out1 = reshape88_out1.permute([0, 2, 1, 3]);
        let reshape89_out1 = transpose66_out1.reshape([1284, 101, 64]);
        let layernormalization22_out1 = {
            self.layernormalization22
                .forward(reshape89_out1.clone())
        };
        let transpose67_out1 = layernormalization22_out1.permute([1, 0, 2]);
        let reshape90_out1 = transpose67_out1.reshape([-1, 64]);
        let linear27_out1 = self.linear27.forward(reshape90_out1);
        let reshape91_out1 = linear27_out1.reshape([101, 1284, 3, 64]);
        let unsqueeze11_out1: Tensor<B, 5> = reshape91_out1.unsqueeze_dims::<5>(&[0]);
        let transpose68_out1 = unsqueeze11_out1.permute([3, 1, 2, 0, 4]);
        let squeeze8_out1 = transpose68_out1.squeeze_dims::<4>(&[-2]);
        let gather22_out1 = {
            let sliced = squeeze8_out1.clone().slice(s![0, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather23_out1 = {
            let sliced = squeeze8_out1.clone().slice(s![1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather24_out1 = {
            let sliced = squeeze8_out1.slice(s![2, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let reshape92_out1 = gather22_out1.reshape([101, 5136, 16]);
        let reshape93_out1 = gather23_out1.reshape([101, 5136, 16]);
        let reshape94_out1 = gather24_out1.reshape([101, 5136, 16]);
        let transpose69_out1 = reshape92_out1.permute([1, 0, 2]);
        let transpose70_out1 = reshape94_out1.permute([1, 0, 2]);
        let transpose71_out1 = reshape93_out1.permute([1, 2, 0]);
        let mul10_out1 = transpose69_out1
            .mul((constant60_out1).unsqueeze_dims(&[0isize, 1isize]));
        let matmul17_out1 = mul10_out1.matmul(transpose71_out1);
        let softmax9_out1 = burn::tensor::activation::softmax(matmul17_out1, 2);
        let matmul18_out1 = softmax9_out1.matmul(transpose70_out1);
        let transpose72_out1 = matmul18_out1.permute([1, 0, 2]);
        let reshape95_out1 = transpose72_out1.reshape([129684, 64]);
        let linear28_out1 = self.linear28.forward(reshape95_out1);
        let reshape96_out1 = linear28_out1.reshape([101, 1284, 64]);
        let transpose73_out1 = reshape96_out1.permute([1, 0, 2]);
        let add23_out1 = reshape89_out1.clone().add(transpose73_out1);
        let layernormalization23_out1 = {
            self.layernormalization23
                .forward(add23_out1.clone())
        };
        let constant190_out1 = self.constant190.val();
        let gru8_out1 = {
            let (output_seq, _final_state) =
                fast_bigru_forward(&self.gru8, layernormalization23_out1, constant190_out1);
            {
                let [seq_len, batch_size, _] = output_seq.dims();
                let reshaped = output_seq.reshape([seq_len, batch_size, 2, 128usize]);
                reshaped.swap_dims(1, 2)
            }
        };
        let transpose74_out1 = gru8_out1.permute([0, 2, 1, 3]);
        let reshape97_out1 = transpose74_out1.reshape([1284, 101, 256]);
        let leakyrelu8_out1 = burn::tensor::activation::leaky_relu(
            reshape97_out1,
            0.009999999776482582,
        );
        let reshape98_out1 = leakyrelu8_out1.reshape([-1, 256]);
        let linear29_out1 = self.linear29.forward(reshape98_out1);
        let reshape99_out1 = linear29_out1.reshape([1284, 101, 64]);
        let add24_out1 = add23_out1.add(reshape99_out1);
        let layernormalization24_out1 = {
            self.layernormalization24
                .forward(add24_out1)
        };
        let add25_out1 = layernormalization24_out1.add(reshape89_out1);
        add25_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4<B: Backend> {
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    instancenormalization7: InstanceNorm<B>,
    instancenormalization8: InstanceNorm<B>,
    prelu7: PRelu<B>,
    prelu8: PRelu<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    instancenormalization9: InstanceNorm<B>,
    instancenormalization10: InstanceNorm<B>,
    prelu9: PRelu<B>,
    prelu10: PRelu<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    instancenormalization11: InstanceNorm<B>,
    instancenormalization12: InstanceNorm<B>,
    prelu11: PRelu<B>,
    prelu12: PRelu<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    instancenormalization13: InstanceNorm<B>,
    instancenormalization14: InstanceNorm<B>,
    prelu13: PRelu<B>,
    prelu14: PRelu<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    instancenormalization15: InstanceNorm<B>,
    instancenormalization16: InstanceNorm<B>,
    prelu15: PRelu<B>,
    prelu16: PRelu<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    conv2d19: Conv2d<B>,
    constant254: burn::module::Param<Tensor<B, 2>>,
    constant255: burn::module::Param<Tensor<B, 1>>,
    constant256: burn::module::Param<Tensor<B, 1>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule4<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d7 = Conv2dConfig::new([64, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([64, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization7 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let instancenormalization8 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu7 = PReluConfig::new().with_num_parameters(64).init(device);
        let prelu8 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d9 = Conv2dConfig::new([128, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 1, 0, 1))
            .with_dilation([2, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([128, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 1, 0, 1))
            .with_dilation([2, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization9 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let instancenormalization10 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu9 = PReluConfig::new().with_num_parameters(64).init(device);
        let prelu10 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d11 = Conv2dConfig::new([192, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 1, 0, 1))
            .with_dilation([4, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([192, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 1, 0, 1))
            .with_dilation([4, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization11 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let instancenormalization12 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu11 = PReluConfig::new().with_num_parameters(64).init(device);
        let prelu12 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d13 = Conv2dConfig::new([256, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(8, 1, 0, 1))
            .with_dilation([8, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([256, 64], [2, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(8, 1, 0, 1))
            .with_dilation([8, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization13 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let instancenormalization14 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu13 = PReluConfig::new().with_num_parameters(64).init(device);
        let prelu14 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d15 = Conv2dConfig::new([64, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([64, 128], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization15 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let instancenormalization16 = InstanceNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let prelu15 = PReluConfig::new().with_num_parameters(64).init(device);
        let prelu16 = PReluConfig::new().with_num_parameters(64).init(device);
        let conv2d17 = Conv2dConfig::new([64, 1], [1, 2])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([64, 1], [1, 2])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([64, 1], [1, 2])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant254: burn::module::Param<Tensor<B, 2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                2,
            >::zeros([201, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [201, 1].into(),
        );
        let constant255: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data(
                burn::tensor::TensorData::from([3.1415927410125732f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant256: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data(
                burn::tensor::TensorData::from([2f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        Self {
            conv2d7,
            conv2d8,
            instancenormalization7,
            instancenormalization8,
            prelu7,
            prelu8,
            conv2d9,
            conv2d10,
            instancenormalization9,
            instancenormalization10,
            prelu9,
            prelu10,
            conv2d11,
            conv2d12,
            instancenormalization11,
            instancenormalization12,
            prelu11,
            prelu12,
            conv2d13,
            conv2d14,
            instancenormalization13,
            instancenormalization14,
            prelu13,
            prelu14,
            conv2d15,
            conv2d16,
            instancenormalization15,
            instancenormalization16,
            prelu15,
            prelu16,
            conv2d17,
            conv2d18,
            conv2d19,
            constant254,
            constant255,
            constant256,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add25_out1: Tensor<B, 3>,
        noisy_amp: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let reshape100_out1 = add25_out1.reshape([4, 321, 101, 64]);
        let transpose75_out1 = reshape100_out1.permute([0, 3, 1, 2]);
        let conv2d7_out1 = self.conv2d7.forward(transpose75_out1.clone());
        let conv2d8_out1 = self.conv2d8.forward(transpose75_out1.clone());
        let instancenormalization7_out1 = self
            .instancenormalization7
            .forward(conv2d7_out1);
        let instancenormalization8_out1 = self
            .instancenormalization8
            .forward(conv2d8_out1);
        let prelu7_out1 = self.prelu7.forward(instancenormalization7_out1);
        let prelu8_out1 = self.prelu8.forward(instancenormalization8_out1);
        let concat6_out1 = burn::tensor::Tensor::cat(
            [prelu7_out1, transpose75_out1.clone()].into(),
            1,
        );
        let concat7_out1 = burn::tensor::Tensor::cat(
            [prelu8_out1, transpose75_out1].into(),
            1,
        );
        let conv2d9_out1 = self.conv2d9.forward(concat6_out1.clone());
        let conv2d10_out1 = self.conv2d10.forward(concat7_out1.clone());
        let instancenormalization9_out1 = self
            .instancenormalization9
            .forward(conv2d9_out1);
        let instancenormalization10_out1 = self
            .instancenormalization10
            .forward(conv2d10_out1);
        let prelu9_out1 = self.prelu9.forward(instancenormalization9_out1);
        let prelu10_out1 = self.prelu10.forward(instancenormalization10_out1);
        let concat8_out1 = burn::tensor::Tensor::cat(
            [prelu9_out1, concat6_out1].into(),
            1,
        );
        let concat9_out1 = burn::tensor::Tensor::cat(
            [prelu10_out1, concat7_out1].into(),
            1,
        );
        let conv2d11_out1 = self.conv2d11.forward(concat8_out1.clone());
        let conv2d12_out1 = self.conv2d12.forward(concat9_out1.clone());
        let instancenormalization11_out1 = self
            .instancenormalization11
            .forward(conv2d11_out1);
        let instancenormalization12_out1 = self
            .instancenormalization12
            .forward(conv2d12_out1);
        let prelu11_out1 = self.prelu11.forward(instancenormalization11_out1);
        let prelu12_out1 = self.prelu12.forward(instancenormalization12_out1);
        let concat10_out1 = burn::tensor::Tensor::cat(
            [prelu11_out1, concat8_out1].into(),
            1,
        );
        let concat11_out1 = burn::tensor::Tensor::cat(
            [prelu12_out1, concat9_out1].into(),
            1,
        );
        let conv2d13_out1 = self.conv2d13.forward(concat10_out1);
        let conv2d14_out1 = self.conv2d14.forward(concat11_out1);
        let instancenormalization13_out1 = self
            .instancenormalization13
            .forward(conv2d13_out1);
        let instancenormalization14_out1 = self
            .instancenormalization14
            .forward(conv2d14_out1);
        let prelu13_out1 = self.prelu13.forward(instancenormalization13_out1);
        let prelu14_out1 = self.prelu14.forward(instancenormalization14_out1);
        let conv2d15_out1 = self.conv2d15.forward(prelu13_out1);
        let conv2d16_out1 = self.conv2d16.forward(prelu14_out1);
        let reshape101_out1 = conv2d15_out1.reshape([4, 2, 64, 321, 101]);
        let reshape102_out1 = conv2d16_out1.reshape([4, 2, 64, 321, 101]);
        let transpose76_out1 = reshape101_out1.permute([0, 2, 3, 4, 1]);
        let transpose77_out1 = reshape102_out1.permute([0, 2, 3, 4, 1]);
        let reshape103_out1 = transpose76_out1.reshape([4, 64, 321, -1]);
        let reshape104_out1 = transpose77_out1.reshape([4, 64, 321, -1]);
        let instancenormalization15_out1 = self
            .instancenormalization15
            .forward(reshape103_out1);
        let instancenormalization16_out1 = self
            .instancenormalization16
            .forward(reshape104_out1);
        let prelu15_out1 = self.prelu15.forward(instancenormalization15_out1);
        let prelu16_out1 = self.prelu16.forward(instancenormalization16_out1);
        let conv2d17_out1 = self.conv2d17.forward(prelu15_out1);
        let conv2d18_out1 = self.conv2d18.forward(prelu16_out1.clone());
        let conv2d19_out1 = self.conv2d19.forward(prelu16_out1);
        let transpose78_out1 = conv2d17_out1.permute([0, 3, 2, 1]);
        let div3_out1 = conv2d19_out1.clone().div(conv2d18_out1.clone());
        let constant253_out1 = 0f32;
        let greater1_out1 = conv2d19_out1.greater_elem(constant253_out1);
        let less1_out1 = conv2d18_out1.lower_elem(constant253_out1);
        let squeeze9_out1 = transpose78_out1.squeeze_dims::<3>(&[-1]);
        let atan1_out1 = div3_out1.atan();
        let constant254_out1 = self.constant254.val();
        let mul11_out1 = (constant254_out1).unsqueeze_dims(&[0isize]).mul(squeeze9_out1);
        let constant255_out1 = self.constant255.val();
        let add26_out1 = atan1_out1
            .clone()
            .add((constant255_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let sub2_out1 = atan1_out1
            .clone()
            .sub((constant255_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(mul11_out1);
        let where1_out1 = sub2_out1.mask_where(greater1_out1, add26_out1);
        let constant256_out1 = self.constant256.val();
        let mul12_out1 = sigmoid2_out1
            .mul((constant256_out1).unsqueeze_dims(&[0isize, 1isize]));
        let where2_out1 = atan1_out1.mask_where(less1_out1, where1_out1);
        let mul13_out1 = noisy_amp.mul(mul12_out1);
        let isnan1_out1 = where2_out1.clone().is_nan();
        let where3_out1 = where2_out1.mask_fill(isnan1_out1, constant253_out1);
        let transpose79_out1 = where3_out1.permute([0, 3, 2, 1]);
        let squeeze10_out1 = transpose79_out1.squeeze_dims::<3>(&[-1]);
        (mul13_out1, squeeze10_out1)
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    submodule1: Submodule1<B>,
    submodule2: Submodule2<B>,
    submodule3: Submodule3<B>,
    submodule4: Submodule4<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}


extern crate std;

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file(
            "/tmp/burn_onnx_p8hem3sc/target/debug/build/burn-onnx-converter-g-map-se-b4-b10040af164176b3/out/model/g_map_se_b4.bpk",
            &Default::default(),
        )
    }
}

impl<B: Backend> Model<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model.load_from(&mut store).expect("Failed to load burnpack file");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model.load_from(&mut store).expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let submodule1 = Submodule1::new(device);
        let submodule2 = Submodule2::new(device);
        let submodule3 = Submodule3::new(device);
        let submodule4 = Submodule4::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            submodule4,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        noisy_amp: Tensor<B, 3>,
        noisy_pha: Tensor<B, 3>,
        prior_embedding: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let (add4_out1, constant60_out1) = self
            .submodule1
            .forward(noisy_amp.clone(), noisy_pha, prior_embedding);
        let add10_out1 = self.submodule2.forward(add4_out1, constant60_out1.clone());
        let add25_out1 = self.submodule3.forward(add10_out1, constant60_out1);
        let (mul13_out1, squeeze10_out1) = self
            .submodule4
            .forward(add25_out1, noisy_amp);
        (mul13_out1, squeeze10_out1)
    }
}
