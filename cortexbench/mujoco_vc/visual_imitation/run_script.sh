function run_diffusion {
    echo "Running with config: $SPECIAL_ARGS"
    python hydra_launcher.py --config-name $CONFIG_NAME \
        env=hammer-v2-goal-observable \
        hydra/launcher=slurm \
        $SPECIAL_ARGS \
        diffusion_prompt=""

    python hydra_launcher.py --config-name $CONFIG_NAME \
        env=drawer-open-v2-goal-observable \
        hydra/launcher=slurm \
        $SPECIAL_ARGS \
        diffusion_prompt=""

    python hydra_launcher.py --config-name $CONFIG_NAME \
        env=bin-picking-v2-goal-observable \
        hydra/launcher=slurm \
        $SPECIAL_ARGS \
        diffusion_prompt=""

    python hydra_launcher.py --config-name $CONFIG_NAME \
        env=button-press-topdown-v2-goal-observable \
        hydra/launcher=slurm \
        $SPECIAL_ARGS \
        diffusion_prompt=""

    python hydra_launcher.py --config-name $CONFIG_NAME \
        env=assembly-v2-goal-observable \
        hydra/launcher=slurm \
        $SPECIAL_ARGS \
        diffusion_prompt=""
}

CONFIG_NAME=Metaworld_BC_config.yaml

SPECIAL_ARGS="model=diffusion_sd_15_laion \
    spatial_dims=8 \
    embed_channel_dim=4480 \
    reduced_channel_dim=12 \
    emb_consolidation_type=3x3_conv \
    diffusion_timesteps=[200] \
    model.model.representation_layer_name=["down_1","down_2","down_3","mid"] \
    model.model.unet_path=/path/to/unet"

SPECIAL_ARGS="$SPECIAL_ARGS seed=12345"
run_diffusion

SPECIAL_ARGS="$SPECIAL_ARGS seed=23451"
run_diffusion

SPECIAL_ARGS="$SPECIAL_ARGS seed=34512"
run_diffusion