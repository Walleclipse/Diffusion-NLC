import argparse
import inspect

#from . import gaussian_diffusion as gd
#from .respace import SpacedDiffusion, space_timesteps
from .unet_adm import  UNetModel, SigmaModel, EncoderUNetModel
from .unet_simple import Model as UnetSimple
from .unet_simple import SigmaModel as SigmaSimple
from .edm_networks import SongUNet
from .edm_networks import SigmaModel as SigmaEDM

NUM_CLASSES = 1000


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res



def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_sigma_eps_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0.0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    sigma_block=2,
    sigma_dropout=0.0,
    use_sigma_fp16=False,
    **kwargs
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    eps_model = UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

    inp_channels = int(num_channels * channel_mult[-1])
    inp_dim = int(image_size * 0.5 ** (len(channel_mult) - 1))
    sigma_model = SigmaModel(dim=inp_dim, channels=inp_channels, n_blocks=sigma_block, out_dim=1, dropout=sigma_dropout,
                             num_heads=num_heads, num_head_channels=num_head_channels,
                             use_new_attention_order=use_new_attention_order,
                             use_checkpoint=use_checkpoint, use_fp16=use_sigma_fp16)
    feat_shape = (inp_channels,inp_dim,inp_dim )

    return eps_model, sigma_model,feat_shape


def create_simple_sigma_eps_model(config):
    eps_model = UnetSimple(config)

    num_channels, out_ch, channel_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)

    inp_channels = int(num_channels * channel_mult[-1])
    inp_dim = int(config.data.image_size * 0.5 ** (len(channel_mult) - 1))
    sigma_model = SigmaSimple( dim=inp_dim, channels=inp_channels, n_blocks=config.model.sigma_block, out_dim=1,dropout=config.model.sigma_dropout)
    feat_shape = (inp_channels,inp_dim,inp_dim )

    return eps_model, sigma_model,feat_shape


def create_edm_sigma_eps_model(
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.

        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 2, 2],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.

        embedding_type='positional',  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        encoder_type='standard',  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type='standard',  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[1, 1],
        sigma_block=2,
        sigma_dropout=0.0,
        **kwargs
):
    eps_model = SongUNet(
        img_resolution=img_resolution,  # Image resolution at input/output.
        in_channels=in_channels,  # Number of color channels at input.
        out_channels=out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=augment_dim,  # Augmentation label dimensionality, 0 = no augmentation.

        model_channels=model_channels,  # Base multiplier for the number of channels.
        channel_mult=channel_mult,  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=channel_mult_emb,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=num_blocks,  # Number of residual blocks per resolution.
        attn_resolutions=attn_resolutions,  # List of resolutions with self-attention.
        dropout=dropout,  # Dropout probability of intermediate activations.

        embedding_type=embedding_type,  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type=encoder_type,  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type=decoder_type,  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=resample_filter,
    )

    inp_channels = int(model_channels * channel_mult[-1])
    inp_dim = int(img_resolution * 0.5 ** (len(channel_mult) - 1))
    sigma_model = SigmaEDM(dim=inp_dim, channels=inp_channels, n_blocks=sigma_block, out_dim=1, dropout=sigma_dropout,
                             resample_filter = resample_filter)
    feat_shape = (inp_channels, inp_dim, inp_dim)

    return eps_model, sigma_model, feat_shape


def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")