// Generated from ONNX "model/voxceleb_ecapa512.onnx" by burn-onnx
use burn::prelude::*;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LinearLayout;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::Conv1d;
use burn::nn::conv::Conv1dConfig;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    constant185: burn::module::Param<Tensor<B, 1>>,
    constant187: burn::module::Param<Tensor<B, 1>>,
    constant188: burn::module::Param<Tensor<B, 1>>,
    constant189: burn::module::Param<Tensor<B, 1>>,
    conv1d1: Conv1d<B>,
    batchnormalization1: BatchNorm<B>,
    conv1d2: Conv1d<B>,
    batchnormalization2: BatchNorm<B>,
    conv1d3: Conv1d<B>,
    batchnormalization3: BatchNorm<B>,
    conv1d4: Conv1d<B>,
    batchnormalization4: BatchNorm<B>,
    conv1d5: Conv1d<B>,
    batchnormalization5: BatchNorm<B>,
    conv1d6: Conv1d<B>,
    batchnormalization6: BatchNorm<B>,
    conv1d7: Conv1d<B>,
    batchnormalization7: BatchNorm<B>,
    conv1d8: Conv1d<B>,
    batchnormalization8: BatchNorm<B>,
    conv1d9: Conv1d<B>,
    batchnormalization9: BatchNorm<B>,
    conv1d10: Conv1d<B>,
    batchnormalization10: BatchNorm<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    conv1d11: Conv1d<B>,
    batchnormalization11: BatchNorm<B>,
    conv1d12: Conv1d<B>,
    batchnormalization12: BatchNorm<B>,
    conv1d13: Conv1d<B>,
    batchnormalization13: BatchNorm<B>,
    conv1d14: Conv1d<B>,
    batchnormalization14: BatchNorm<B>,
    conv1d15: Conv1d<B>,
    batchnormalization15: BatchNorm<B>,
    conv1d16: Conv1d<B>,
    batchnormalization16: BatchNorm<B>,
    conv1d17: Conv1d<B>,
    batchnormalization17: BatchNorm<B>,
    conv1d18: Conv1d<B>,
    batchnormalization18: BatchNorm<B>,
    conv1d19: Conv1d<B>,
    batchnormalization19: BatchNorm<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    conv1d20: Conv1d<B>,
    batchnormalization20: BatchNorm<B>,
    conv1d21: Conv1d<B>,
    batchnormalization21: BatchNorm<B>,
    conv1d22: Conv1d<B>,
    batchnormalization22: BatchNorm<B>,
    conv1d23: Conv1d<B>,
    batchnormalization23: BatchNorm<B>,
    conv1d24: Conv1d<B>,
    batchnormalization24: BatchNorm<B>,
    conv1d25: Conv1d<B>,
    batchnormalization25: BatchNorm<B>,
    conv1d26: Conv1d<B>,
    batchnormalization26: BatchNorm<B>,
    conv1d27: Conv1d<B>,
    batchnormalization27: BatchNorm<B>,
    conv1d28: Conv1d<B>,
    batchnormalization28: BatchNorm<B>,
    linear5: Linear<B>,
    linear6: Linear<B>,
    conv1d29: Conv1d<B>,
    conv1d30: Conv1d<B>,
    conv1d31: Conv1d<B>,
    batchnormalization29: BatchNorm<B>,
    linear7: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}


extern crate std;

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file(
            "/tmp/burn_onnx_zqj9wdb7/target/debug/build/burn-onnx-converter-voxceleb-ecapa512-c337722ddd1ee63b/out/model/voxceleb_ecapa512.bpk",
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
        let constant185: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
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
        let constant187: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data(
                burn::tensor::TensorData::from([321f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant188: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data(
                burn::tensor::TensorData::from([320f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant189: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.00000010000000116860974f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let conv1d1 = Conv1dConfig::new(80, 512, 5)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2, 2))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization1 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d2 = Conv1dConfig::new(512, 512, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d3 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2, 2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization3 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d4 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2, 2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization4 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d5 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2, 2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization5 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d6 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2, 2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization6 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d7 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2, 2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization7 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d8 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2, 2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization8 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d9 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2, 2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization9 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d10 = Conv1dConfig::new(512, 512, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization10 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let linear1 = LinearConfig::new(512, 128)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear2 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv1d11 = Conv1dConfig::new(512, 512, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization11 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d12 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .with_dilation(3)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization12 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d13 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .with_dilation(3)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization13 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d14 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .with_dilation(3)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization14 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d15 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .with_dilation(3)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization15 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d16 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .with_dilation(3)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization16 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d17 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .with_dilation(3)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization17 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d18 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .with_dilation(3)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization18 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d19 = Conv1dConfig::new(512, 512, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization19 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let linear3 = LinearConfig::new(512, 128)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear4 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv1d20 = Conv1dConfig::new(512, 512, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization20 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d21 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(4, 4))
            .with_dilation(4)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization21 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d22 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(4, 4))
            .with_dilation(4)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization22 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d23 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(4, 4))
            .with_dilation(4)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization23 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d24 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(4, 4))
            .with_dilation(4)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization24 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d25 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(4, 4))
            .with_dilation(4)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization25 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d26 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(4, 4))
            .with_dilation(4)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization26 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d27 = Conv1dConfig::new(64, 64, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(4, 4))
            .with_dilation(4)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization27 = BatchNormConfig::new(64)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d28 = Conv1dConfig::new(512, 512, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization28 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let linear5 = LinearConfig::new(512, 128)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear6 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv1d29 = Conv1dConfig::new(1536, 1536, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d30 = Conv1dConfig::new(4608, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d31 = Conv1dConfig::new(128, 1536, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization29 = BatchNormConfig::new(3072)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let linear7 = LinearConfig::new(3072, 192)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            constant185,
            constant187,
            constant188,
            constant189,
            conv1d1,
            batchnormalization1,
            conv1d2,
            batchnormalization2,
            conv1d3,
            batchnormalization3,
            conv1d4,
            batchnormalization4,
            conv1d5,
            batchnormalization5,
            conv1d6,
            batchnormalization6,
            conv1d7,
            batchnormalization7,
            conv1d8,
            batchnormalization8,
            conv1d9,
            batchnormalization9,
            conv1d10,
            batchnormalization10,
            linear1,
            linear2,
            conv1d11,
            batchnormalization11,
            conv1d12,
            batchnormalization12,
            conv1d13,
            batchnormalization13,
            conv1d14,
            batchnormalization14,
            conv1d15,
            batchnormalization15,
            conv1d16,
            batchnormalization16,
            conv1d17,
            batchnormalization17,
            conv1d18,
            batchnormalization18,
            conv1d19,
            batchnormalization19,
            linear3,
            linear4,
            conv1d20,
            batchnormalization20,
            conv1d21,
            batchnormalization21,
            conv1d22,
            batchnormalization22,
            conv1d23,
            batchnormalization23,
            conv1d24,
            batchnormalization24,
            conv1d25,
            batchnormalization25,
            conv1d26,
            batchnormalization26,
            conv1d27,
            batchnormalization27,
            conv1d28,
            batchnormalization28,
            linear5,
            linear6,
            conv1d29,
            conv1d30,
            conv1d31,
            batchnormalization29,
            linear7,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, feats: Tensor<B, 3>) -> Tensor<B, 2> {
        let constant185_out1 = self.constant185.val();
        let constant187_out1 = self.constant187.val();
        let constant188_out1 = self.constant188.val();
        let constant189_out1 = self.constant189.val();
        let transpose1_out1 = feats.permute([0, 2, 1]);
        let conv1d1_out1 = self.conv1d1.forward(transpose1_out1);
        let relu1_out1 = burn::tensor::activation::relu(conv1d1_out1);
        let batchnormalization1_out1 = self.batchnormalization1.forward(relu1_out1);
        let conv1d2_out1 = self.conv1d2.forward(batchnormalization1_out1.clone());
        let relu2_out1 = burn::tensor::activation::relu(conv1d2_out1);
        let batchnormalization2_out1 = self.batchnormalization2.forward(relu2_out1);
        let split_tensors = batchnormalization2_out1
            .split_with_sizes([64, 64, 64, 64, 64, 64, 64, 64].into(), 1);
        let [split1_out1, split1_out2, split1_out3, split1_out4, split1_out5,
        split1_out6, split1_out7, split1_out8] = split_tensors.try_into().unwrap();
        let conv1d3_out1 = self.conv1d3.forward(split1_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv1d3_out1);
        let batchnormalization3_out1 = self.batchnormalization3.forward(relu3_out1);
        let add1_out1 = batchnormalization3_out1.clone().add(split1_out2);
        let conv1d4_out1 = self.conv1d4.forward(add1_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv1d4_out1);
        let batchnormalization4_out1 = self.batchnormalization4.forward(relu4_out1);
        let add2_out1 = batchnormalization4_out1.clone().add(split1_out3);
        let conv1d5_out1 = self.conv1d5.forward(add2_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv1d5_out1);
        let batchnormalization5_out1 = self.batchnormalization5.forward(relu5_out1);
        let add3_out1 = batchnormalization5_out1.clone().add(split1_out4);
        let conv1d6_out1 = self.conv1d6.forward(add3_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv1d6_out1);
        let batchnormalization6_out1 = self.batchnormalization6.forward(relu6_out1);
        let add4_out1 = batchnormalization6_out1.clone().add(split1_out5);
        let conv1d7_out1 = self.conv1d7.forward(add4_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv1d7_out1);
        let batchnormalization7_out1 = self.batchnormalization7.forward(relu7_out1);
        let add5_out1 = batchnormalization7_out1.clone().add(split1_out6);
        let conv1d8_out1 = self.conv1d8.forward(add5_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv1d8_out1);
        let batchnormalization8_out1 = self.batchnormalization8.forward(relu8_out1);
        let add6_out1 = batchnormalization8_out1.clone().add(split1_out7);
        let conv1d9_out1 = self.conv1d9.forward(add6_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv1d9_out1);
        let batchnormalization9_out1 = self.batchnormalization9.forward(relu9_out1);
        let concat1_out1 = burn::tensor::Tensor::cat(
            [
                batchnormalization3_out1,
                batchnormalization4_out1,
                batchnormalization5_out1,
                batchnormalization6_out1,
                batchnormalization7_out1,
                batchnormalization8_out1,
                batchnormalization9_out1,
                split1_out8,
            ]
                .into(),
            1,
        );
        let conv1d10_out1 = self.conv1d10.forward(concat1_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv1d10_out1);
        let batchnormalization10_out1 = self.batchnormalization10.forward(relu10_out1);
        let reducemean1_out1 = {
            batchnormalization10_out1
                .clone()
                .mean_dim(2usize)
                .squeeze_dims::<2usize>(&[2])
        };
        let linear1_out1 = self.linear1.forward(reducemean1_out1);
        let relu11_out1 = burn::tensor::activation::relu(linear1_out1);
        let linear2_out1 = self.linear2.forward(relu11_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(linear2_out1);
        let unsqueeze1_out1: Tensor<B, 3> = sigmoid1_out1.unsqueeze_dims::<3>(&[2]);
        let mul1_out1 = batchnormalization10_out1.mul(unsqueeze1_out1);
        let add7_out1 = batchnormalization1_out1.add(mul1_out1);
        let conv1d11_out1 = self.conv1d11.forward(add7_out1.clone());
        let relu12_out1 = burn::tensor::activation::relu(conv1d11_out1);
        let batchnormalization11_out1 = self.batchnormalization11.forward(relu12_out1);
        let split_tensors = batchnormalization11_out1
            .split_with_sizes([64, 64, 64, 64, 64, 64, 64, 64].into(), 1);
        let [split2_out1, split2_out2, split2_out3, split2_out4, split2_out5,
        split2_out6, split2_out7, split2_out8] = split_tensors.try_into().unwrap();
        let conv1d12_out1 = self.conv1d12.forward(split2_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv1d12_out1);
        let batchnormalization12_out1 = self.batchnormalization12.forward(relu13_out1);
        let add8_out1 = batchnormalization12_out1.clone().add(split2_out2);
        let conv1d13_out1 = self.conv1d13.forward(add8_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv1d13_out1);
        let batchnormalization13_out1 = self.batchnormalization13.forward(relu14_out1);
        let add9_out1 = batchnormalization13_out1.clone().add(split2_out3);
        let conv1d14_out1 = self.conv1d14.forward(add9_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv1d14_out1);
        let batchnormalization14_out1 = self.batchnormalization14.forward(relu15_out1);
        let add10_out1 = batchnormalization14_out1.clone().add(split2_out4);
        let conv1d15_out1 = self.conv1d15.forward(add10_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv1d15_out1);
        let batchnormalization15_out1 = self.batchnormalization15.forward(relu16_out1);
        let add11_out1 = batchnormalization15_out1.clone().add(split2_out5);
        let conv1d16_out1 = self.conv1d16.forward(add11_out1);
        let relu17_out1 = burn::tensor::activation::relu(conv1d16_out1);
        let batchnormalization16_out1 = self.batchnormalization16.forward(relu17_out1);
        let add12_out1 = batchnormalization16_out1.clone().add(split2_out6);
        let conv1d17_out1 = self.conv1d17.forward(add12_out1);
        let relu18_out1 = burn::tensor::activation::relu(conv1d17_out1);
        let batchnormalization17_out1 = self.batchnormalization17.forward(relu18_out1);
        let add13_out1 = batchnormalization17_out1.clone().add(split2_out7);
        let conv1d18_out1 = self.conv1d18.forward(add13_out1);
        let relu19_out1 = burn::tensor::activation::relu(conv1d18_out1);
        let batchnormalization18_out1 = self.batchnormalization18.forward(relu19_out1);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [
                batchnormalization12_out1,
                batchnormalization13_out1,
                batchnormalization14_out1,
                batchnormalization15_out1,
                batchnormalization16_out1,
                batchnormalization17_out1,
                batchnormalization18_out1,
                split2_out8,
            ]
                .into(),
            1,
        );
        let conv1d19_out1 = self.conv1d19.forward(concat2_out1);
        let relu20_out1 = burn::tensor::activation::relu(conv1d19_out1);
        let batchnormalization19_out1 = self.batchnormalization19.forward(relu20_out1);
        let reducemean2_out1 = {
            batchnormalization19_out1
                .clone()
                .mean_dim(2usize)
                .squeeze_dims::<2usize>(&[2])
        };
        let linear3_out1 = self.linear3.forward(reducemean2_out1);
        let relu21_out1 = burn::tensor::activation::relu(linear3_out1);
        let linear4_out1 = self.linear4.forward(relu21_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(linear4_out1);
        let unsqueeze2_out1: Tensor<B, 3> = sigmoid2_out1.unsqueeze_dims::<3>(&[2]);
        let mul2_out1 = batchnormalization19_out1.mul(unsqueeze2_out1);
        let add14_out1 = add7_out1.clone().add(mul2_out1);
        let conv1d20_out1 = self.conv1d20.forward(add14_out1.clone());
        let relu22_out1 = burn::tensor::activation::relu(conv1d20_out1);
        let batchnormalization20_out1 = self.batchnormalization20.forward(relu22_out1);
        let split_tensors = batchnormalization20_out1
            .split_with_sizes([64, 64, 64, 64, 64, 64, 64, 64].into(), 1);
        let [split3_out1, split3_out2, split3_out3, split3_out4, split3_out5,
        split3_out6, split3_out7, split3_out8] = split_tensors.try_into().unwrap();
        let conv1d21_out1 = self.conv1d21.forward(split3_out1);
        let relu23_out1 = burn::tensor::activation::relu(conv1d21_out1);
        let batchnormalization21_out1 = self.batchnormalization21.forward(relu23_out1);
        let add15_out1 = batchnormalization21_out1.clone().add(split3_out2);
        let conv1d22_out1 = self.conv1d22.forward(add15_out1);
        let relu24_out1 = burn::tensor::activation::relu(conv1d22_out1);
        let batchnormalization22_out1 = self.batchnormalization22.forward(relu24_out1);
        let add16_out1 = batchnormalization22_out1.clone().add(split3_out3);
        let conv1d23_out1 = self.conv1d23.forward(add16_out1);
        let relu25_out1 = burn::tensor::activation::relu(conv1d23_out1);
        let batchnormalization23_out1 = self.batchnormalization23.forward(relu25_out1);
        let add17_out1 = batchnormalization23_out1.clone().add(split3_out4);
        let conv1d24_out1 = self.conv1d24.forward(add17_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv1d24_out1);
        let batchnormalization24_out1 = self.batchnormalization24.forward(relu26_out1);
        let add18_out1 = batchnormalization24_out1.clone().add(split3_out5);
        let conv1d25_out1 = self.conv1d25.forward(add18_out1);
        let relu27_out1 = burn::tensor::activation::relu(conv1d25_out1);
        let batchnormalization25_out1 = self.batchnormalization25.forward(relu27_out1);
        let add19_out1 = batchnormalization25_out1.clone().add(split3_out6);
        let conv1d26_out1 = self.conv1d26.forward(add19_out1);
        let relu28_out1 = burn::tensor::activation::relu(conv1d26_out1);
        let batchnormalization26_out1 = self.batchnormalization26.forward(relu28_out1);
        let add20_out1 = batchnormalization26_out1.clone().add(split3_out7);
        let conv1d27_out1 = self.conv1d27.forward(add20_out1);
        let relu29_out1 = burn::tensor::activation::relu(conv1d27_out1);
        let batchnormalization27_out1 = self.batchnormalization27.forward(relu29_out1);
        let concat3_out1 = burn::tensor::Tensor::cat(
            [
                batchnormalization21_out1,
                batchnormalization22_out1,
                batchnormalization23_out1,
                batchnormalization24_out1,
                batchnormalization25_out1,
                batchnormalization26_out1,
                batchnormalization27_out1,
                split3_out8,
            ]
                .into(),
            1,
        );
        let conv1d28_out1 = self.conv1d28.forward(concat3_out1);
        let relu30_out1 = burn::tensor::activation::relu(conv1d28_out1);
        let batchnormalization28_out1 = self.batchnormalization28.forward(relu30_out1);
        let reducemean3_out1 = {
            batchnormalization28_out1
                .clone()
                .mean_dim(2usize)
                .squeeze_dims::<2usize>(&[2])
        };
        let linear5_out1 = self.linear5.forward(reducemean3_out1);
        let relu31_out1 = burn::tensor::activation::relu(linear5_out1);
        let linear6_out1 = self.linear6.forward(relu31_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(linear6_out1);
        let unsqueeze3_out1: Tensor<B, 3> = sigmoid3_out1.unsqueeze_dims::<3>(&[2]);
        let mul3_out1 = batchnormalization28_out1.mul(unsqueeze3_out1);
        let add21_out1 = add14_out1.clone().add(mul3_out1);
        let concat4_out1 = burn::tensor::Tensor::cat(
            [add7_out1, add14_out1, add21_out1].into(),
            1,
        );
        let conv1d29_out1 = self.conv1d29.forward(concat4_out1);
        let relu32_out1 = burn::tensor::activation::relu(conv1d29_out1);
        let reducemean4_out1 = { relu32_out1.clone().mean_dim(2usize) };
        let pow1_out1 = relu32_out1
            .clone()
            .powf((constant185_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let expand1_out1 = {
            let onnx_shape: [i64; 3usize] = [1, 1536, 321];
            let input_dims = reducemean4_out1.clone().dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..3usize {
                let dim_offset = 3usize - 3usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            reducemean4_out1.clone().expand(shape)
        };
        let sub1_out1 = relu32_out1.clone().sub(reducemean4_out1);
        let mul4_out1 = sub1_out1.clone().mul(sub1_out1);
        let reducemean5_out1 = { mul4_out1.mean_dim(2usize) };
        let mul5_out1 = reducemean5_out1
            .mul((constant187_out1).unsqueeze_dims(&[0isize, 1isize]));
        let div1_out1 = mul5_out1
            .div((constant188_out1).unsqueeze_dims(&[0isize, 1isize]));
        let add22_out1 = div1_out1
            .add((constant189_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt1_out1 = add22_out1.sqrt();
        let expand2_out1 = {
            let onnx_shape: [i64; 3usize] = [1, 1536, 321];
            let input_dims = sqrt1_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..3usize {
                let dim_offset = 3usize - 3usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            sqrt1_out1.expand(shape)
        };
        let concat5_out1 = burn::tensor::Tensor::cat(
            [relu32_out1.clone(), expand1_out1, expand2_out1].into(),
            1,
        );
        let conv1d30_out1 = self.conv1d30.forward(concat5_out1);
        let tanh1_out1 = conv1d30_out1.tanh();
        let conv1d31_out1 = self.conv1d31.forward(tanh1_out1);
        let softmax1_out1 = burn::tensor::activation::softmax(conv1d31_out1, 2);
        let mul6_out1 = softmax1_out1.clone().mul(relu32_out1);
        let mul7_out1 = softmax1_out1.mul(pow1_out1);
        let reducesum1_out1 = { mul6_out1.sum_dim(2usize).squeeze_dims::<2usize>(&[2]) };
        let reducesum2_out1 = { mul7_out1.sum_dim(2usize).squeeze_dims::<2usize>(&[2]) };
        let pow2_out1 = reducesum1_out1
            .clone()
            .powf((constant185_out1).unsqueeze_dims(&[0isize]));
        let sub2_out1 = reducesum2_out1.sub(pow2_out1);
        let clip1_out1 = {
            let __clip_min = 0.00000010000000116860974f64;
            sub2_out1.clamp_min(__clip_min)
        };
        let sqrt2_out1 = clip1_out1.sqrt();
        let concat6_out1 = burn::tensor::Tensor::cat(
            [reducesum1_out1, sqrt2_out1].into(),
            1,
        );
        let batchnormalization29_out1 = self.batchnormalization29.forward(concat6_out1);
        let linear7_out1 = self.linear7.forward(batchnormalization29_out1);
        linear7_out1
    }
}
