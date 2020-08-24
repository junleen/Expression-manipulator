from .base_option import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--n_threads_train', default=4, type=int, help='# threads for loading data')
        self._parser.add_argument('--num_iters_validate', default=2, type=int, help='# batches to use when validating')
        self._parser.add_argument('--print_freq_s', type=int, default=60, help='frequency of showing training results on console')
        self._parser.add_argument('--display_freq_s', type=int, default=300, help='frequency [s] of showing training results on screen')
        self._parser.add_argument('--save_latest_freq_s', type=int, default=3600, help='frequency of saving the latest results')
        self._parser.add_argument('--save_every_epoch', type=int, default=1, help="save for every #save_every_epoch epoch")
        
        self._parser.add_argument('--nepochs_no_decay', type=int, default=30, help='# of epochs at starting learning rate')
        self._parser.add_argument('--nepochs_decay', type=int, default=20, help='# of epochs to linearly decay learning rate to zero')
        self._parser.add_argument('--lr_policy', type=str, default='step', help='type of learning rate decay')

        self._parser.add_argument('--ngf', type=int, default=64, help='num of filters in first convolution layer in G')
        self._parser.add_argument('--ndf', type=int, default=64, help='num of filters in first convolution layer in D')
        self._parser.add_argument('--normtype_G', type=str, default='instancenorm', help='normtype of G')
        self._parser.add_argument('--normtype_D', type=str, default='none', help='normtype of D')
        self._parser.add_argument('--use_sn_G', action="store_true", default=False, help='use spectral norm for G')
        self._parser.add_argument('--use_sn_D', action="store_true", default=False, help='use spectral norm for D')

        self._parser.add_argument('--train_G_every_n_iterations', type=int, default=4, help='train G every n interations')
        self._parser.add_argument('--poses_g_sigma', type=float, default=0.06, help='initial learning rate for adam')
        
        self._parser.add_argument('--lr_G', type=float, default=0.0001, help='initial learning rate for G adam')
        self._parser.add_argument('--G_adam_b1', type=float, default=0.5, help='beta1 for G adam')
        self._parser.add_argument('--G_adam_b2', type=float, default=0.999, help='beta2 for G adam')

        self._parser.add_argument('--lr_D', type=float, default=0.0001, help='initial learning rate for D adam')
        self._parser.add_argument('--D_adam_b1', type=float, default=0.5, help='beta1 for D adam')
        self._parser.add_argument('--D_adam_b2', type=float, default=0.999, help='beta2 for D adam')
        self._parser.add_argument('--lr_decays_to', type=float, default=5e-6, help='learning rate decay terminal value')

        self._parser.add_argument('--lambda_D_prob', type=float, default=10, help='lambda for real/fake discriminator loss')
        self._parser.add_argument('--lambda_D_cond', type=float, default=150, help='lambda for condition discriminator loss')
        self._parser.add_argument('--lambda_D_gp', type=float, default=10, help='lambda gradient penalty loss')
        
        self._parser.add_argument('--lambda_G_fake_cond', type=float, default=150, help='lambda for condition in generator fake image cond loss')
        self._parser.add_argument('--lambda_rec_l1', type=float, default=30, help='lambda self-reconstruction loss')
        self._parser.add_argument('--lambda_cyc_l1', type=float, default=30, help='lambda cycle loss')
        self._parser.add_argument('--lambda_smooth', type=float, default=1e-5, help='lambda smooth loss')

        self.is_train = True
