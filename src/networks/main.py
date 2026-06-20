from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .fmnist_LeNet import FashionMNIST_LeNet, FashionMNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .mlp import MLP, MLP_Autoencoder, MLP_Physical, MLP_Residual, MLP_State_Only
from .vae import VariationalAutoencoder
from .dgm import DeepGenerativeModel, StackedDeepGenerativeModel
# from .transformer import Transformer, Transformer_Autoencoder
from .lstm import LSTM_Net, LSTM_Autoencoder
import setting


def build_network(net_name, ae_net=None):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'mnist_DGM_M2', 'mnist_DGM_M1M2',
                            'fmnist_LeNet', 'fmnist_DGM_M2', 'fmnist_DGM_M1M2',
                            'cifar10_LeNet', 'cifar10_DGM_M2', 'cifar10_DGM_M1M2',
                            'arrhythmia_mlp', 'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                            'thyroid_mlp',
                            'arrhythmia_DGM_M2', 'cardio_DGM_M2', 'satellite_DGM_M2', 'satimage-2_DGM_M2',
                            'shuttle_DGM_M2', 'thyroid_DGM_M2',
                            'transformer','lstm','spoof_mlp',
                            'spoof_mlp_res', 'spoofing_mlp_state_only', 'mlp_alfa', "mlp_pegasus")
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'mnist_DGM_M2':
        net = DeepGenerativeModel([1*28*28, 2, 32, [128, 64]], classifier_net=MNIST_LeNet)

    if net_name == 'mnist_DGM_M1M2':
        net = StackedDeepGenerativeModel([1*28*28, 2, 32, [128, 64]], features=ae_net)

    if net_name == 'fmnist_LeNet':
        net = FashionMNIST_LeNet()

    if net_name == 'fmnist_DGM_M2':
        net = DeepGenerativeModel([1*28*28, 2, 64, [256, 128]], classifier_net=FashionMNIST_LeNet)

    if net_name == 'fmnist_DGM_M1M2':
        net = StackedDeepGenerativeModel([1*28*28, 2, 64, [256, 128]], features=ae_net)

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_DGM_M2':
        net = DeepGenerativeModel([3*32*32, 2, 128, [512, 256]], classifier_net=CIFAR10_LeNet)

    if net_name == 'cifar10_DGM_M1M2':
        net = StackedDeepGenerativeModel([3*32*32, 2, 128, [512, 256]], features=ae_net)

    if net_name == 'arrhythmia_mlp':
        net = MLP(x_dim=274, h_dims=[128, 64], rep_dim=32, bias=False)

    if net_name == 'cardio_mlp':
        net = MLP(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satellite_mlp':
        net = MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satimage-2_mlp':
        net = MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'shuttle_mlp':
        net = MLP(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'thyroid_mlp':
        net = MLP(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)

    if net_name == 'arrhythmia_DGM_M2':
        net = DeepGenerativeModel([274, 2, 32, [128, 64]])

    if net_name == 'cardio_DGM_M2':
        net = DeepGenerativeModel([21, 2, 8, [32, 16]])

    if net_name == 'satellite_DGM_M2':
        net = DeepGenerativeModel([36, 2, 8, [32, 16]])

    if net_name == 'satimage-2_DGM_M2':
        net = DeepGenerativeModel([36, 2, 8, [32, 16]])

    if net_name == 'shuttle_DGM_M2':
        net = DeepGenerativeModel([9, 2, 8, [32, 16]])

    if net_name == 'thyroid_DGM_M2':
        net = DeepGenerativeModel([6, 2, 4, [32, 16]])

    # if net_name == 'transformer':
    #     net = Transformer(8, 2, 64, 512, 8, 6)

    if net_name == 'lstm':
        net = LSTM_Net(input_size=8, rep_dim=64, num_layers=2)
    
    if net_name == 'spoof_mlp':
        net = MLP(x_dim=1200, h_dims=[setting.hd1, setting.hd2], rep_dim=setting.rep, bias=False)

    if net_name == 'spoofing_mlp_state_only':
        net = MLP(x_dim=1200, h_dims=[setting.hd1, setting.hd2], rep_dim=setting.rep, bias=False)

    if net_name == 'spoof_mlp_res':
        net = MLP(x_dim=1200, h_dims=[setting.hd1, setting.hd2], rep_dim=setting.rep, bias=False)

    if net_name == 'mlp_alfa':
        # 25 = window length, 47 = feature channels (must match read_ALFA_single_failure.py output)
        net = MLP(x_dim=40*47, h_dims=[setting.hd1, setting.hd2], rep_dim=setting.rep, bias=False)

    if net_name == 'mlp_pegasus':
        net = MLP(x_dim=20*44, h_dims=[setting.hd1, setting.hd2], rep_dim=setting.rep, bias=False)
    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'mnist_DGM_M1M2',
                            'fmnist_LeNet', 'fmnist_DGM_M1M2',
                            'cifar10_LeNet', 'cifar10_DGM_M1M2',
                            'arrhythmia_mlp', 'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                            'thyroid_mlp',
                            'transformer', 'lstm', 'spoof_mlp')

    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'mnist_DGM_M1M2':
        ae_net = VariationalAutoencoder([1*28*28, 32, [128, 64]])

    if net_name == 'fmnist_LeNet':
        ae_net = FashionMNIST_LeNet_Autoencoder()

    if net_name == 'fmnist_DGM_M1M2':
        ae_net = VariationalAutoencoder([1*28*28, 64, [256, 128]])

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_DGM_M1M2':
        ae_net = VariationalAutoencoder([3*32*32, 128, [512, 256]])

    if net_name == 'arrhythmia_mlp':
        ae_net = MLP_Autoencoder(x_dim=274, h_dims=[128, 64], rep_dim=32, bias=False)

    if net_name == 'cardio_mlp':
        ae_net = MLP_Autoencoder(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satellite_mlp':
        ae_net = MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satimage-2_mlp':
        ae_net = MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'shuttle_mlp':
        ae_net = MLP_Autoencoder(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'thyroid_mlp':
        ae_net = MLP_Autoencoder(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)
    
    # if net_name == 'transformer':
    #     ae_net = Transformer_Autoencoder()
    
    if net_name == 'lstm':
        ae_net = LSTM_Autoencoder(input_size=8, rep_dim=64, num_layers=2, seq_len=100)

    if net_name == 'spoof_mlp':
        ae_net = MLP_Autoencoder(x_dim=1200, h_dims=[setting.hd1, setting.hd2], rep_dim=setting.rep, bias=False)

    return ae_net

def build_network_cats(net_name):
    """Build a CATSModel for use as the CATS baseline backbone.

    Supported names:
      cats_ts2vec_pegasus, cats_mlp_pegasus
      cats_ts2vec_alfa,    cats_mlp_alfa
    """
    from baselines.cats.cats_model import CATSModel

    implemented = ('cats_ts2vec_pegasus', 'cats_mlp_pegasus',
                   'cats_ts2vec_alfa',    'cats_mlp_alfa')
    assert net_name in implemented, f"Unknown CATS net_name '{net_name}'"

    if net_name == 'cats_ts2vec_pegasus':
        return CATSModel(input_size=44, win_size=20,
                         output_size=setting.rep, proj_size=setting.rep // 2,
                         encoder_type='ts2vec')
    if net_name == 'cats_mlp_pegasus':
        return CATSModel(input_size=44, win_size=20,
                         output_size=setting.rep, proj_size=setting.rep // 2,
                         encoder_type='mlp')
    if net_name == 'cats_ts2vec_alfa':
        return CATSModel(input_size=47, win_size=40,
                         output_size=setting.rep, proj_size=setting.rep // 2,
                         encoder_type='ts2vec')
    if net_name == 'cats_mlp_alfa':
        return CATSModel(input_size=47, win_size=40,
                         output_size=setting.rep, proj_size=setting.rep // 2,
                         encoder_type='mlp')


def build_network_physical(net_name):
    implemented_networks = ('spoof_mlp', 'spoof_mlp_res', 'spoofing_mlp_state_only', 'mlp_alfa', "mlp_pegasus")
    print(net_name)
    assert net_name in implemented_networks

    net_physical = None

    if net_name == 'spoof_mlp':
        net_physical = MLP_Physical(x_dim=1200, h_dims=[setting.hd1, setting.hd2], rep_dim=setting.rep, bias=False)
    
    elif net_name == 'spoof_mlp_res':
        net_physical = MLP_Residual(x_dim=1200, h_dims=[setting.hd1, setting.hd2], rep_dim=setting.rep, bias=False)
    
    elif net_name == 'spoofing_mlp_state_only':
        net_physical = MLP_State_Only(x_dim=1200, h_dims=[setting.hd1, setting.hd2], rep_dim=setting.rep, bias=False)

    elif net_name == 'mlp_alfa':
        # 25 = window length, 47 = feature channels (must match read_ALFA_single_failure.py output)
        net_physical = MLP_Physical(x_dim=40*47, h_dims=[setting.hd1, setting.hd2], out_dim=47, rep_dim=setting.rep, bias=False)

    elif net_name == "mlp_pegasus":
        net_physical = MLP_Physical(x_dim=20*44, h_dims=[setting.hd1, setting.hd2], rep_dim=setting.rep, out_dim=44, bias=False)

    else:
        pass

    return net_physical