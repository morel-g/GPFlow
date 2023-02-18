class Case:
    # Time state
    stationary = "stationary"
    continuous_time = "continuous time"
    # Problem type 2d
    two_spirals = "2spirals"
    two_moons = "two_moons"
    moons = "moons"
    cross_gaussians = "cross gaussians"
    swissroll = "swissroll"
    joint_gaussian = "joint_gaussian"
    eight_gaussians = "eight_gaussians"
    conditionnal8gaussians = "conditionnal8gaussians"
    pinwheel = "pinwheel"
    checkerboard = "checkerboard"
    uniform = "uniform"
    circles = "circles"
    n_dim_gaussians = "n_dim_gaussians"
    mnist = "mnist"
    # latent data
    latent_data = "latent_data"
    dsprites = "dsprites"
    chairs = "chairs"
    celeba = "celeba"
    mnist = "mnist"
    # Time discretization method
    euler = "euler"
    euler_explicit = "euler explicit"
    RK4 = "RK4"
    middle_point = "middle point"
    implicit_order1 = "implicit order 1"
    implicit_order2 = "implicit order 2"
    # Implicit method
    fix_point = "fix point"
    anderson = "anderson"
    # Activation function
    tanh = "tanh"
    log_cosh = "log_cosh"
    relu = "relu"
    # GP flow opt type
    train_gp = "train_gp"
    train_nf = "train_nf"
    # Data to train GP flow on
    train_gp_on_data = "train_gp_on_data"
    train_gp_on_gaussian_noise = "train_gp_on_gaussian_noise"
    # Divergence free velocity type
    scalar_func = "scalar func"
    vector_func = "vector_func"
    # Solve method
    standard = "standard"
    deq_model = "deq_model"
    adjoint = "adjoint"
    # Velocity decay
    poly = "poly"
    exp = "exp"
    # Model name
    ffjord = "ffjord"
    cpflow = "cpflow"
    bnaf = "bnaf"
    ot_flow = "ot_flow"
    ode_flow = "ode_flow"
    # Various order
    order_1 = "order_1"
    order_2 = "order_2"
    RK4_space = "RK4_space"
    # Poly type
    incompressible_poly = "incompressible_poly"
    classic_poly = "classic_poly"
    classic_scalar_poly = "classic_scalar_poly"
    # Domain
    circle = "circle"
    rectangle = "rectangle"
    # Euler methods
    penalization = "penalization"
    spectral_method = "spectral_method"
    # AE or VAE
    ae = "ae"
    vae = "vae"
    # Type of model
    nf_model = "nf_model"
    # Latent case
    default_vae = "default_vae"
    apply_nf = "apply_nf"
    apply_gp = "apply_gp"
    # Training case
    toy = "toy"
    latent = "latent"
