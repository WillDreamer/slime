import os
import shutil

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action


def _prune_old_checkpoints(args, rollout_id):
    """Retention: keep checkpoints whose step (rollout_id+1) is a multiple of
    --save-retain-interval forever; keep only the most recent
    non-retained one (rolling window of 1). Pruning is local to the driver's
    --save dir; the bootstrap's S3 sync is additive (no --delete), so we mirror
    the deletion to S3 too when an S3 mirror of --save is configured via env.

    Megatron checkpoint dirs are iter_{iteration:07d} where iteration == the
    rollout_id passed to save(); the latest_checkpointed_iteration.txt tracker
    is never touched (we only delete dirs strictly older than the one just
    written, and never a permanent one)."""
    interval = getattr(args, "save_retain_interval", None)
    if not interval or not args.save:
        return
    save_dir = args.save
    if not os.path.isdir(save_dir):
        return

    def is_permanent(iteration):
        # step == iteration + 1 (rollout_id+1); keep when step % interval == 0.
        return (iteration + 1) % interval == 0

    # Collect existing iter_ dirs and their iteration numbers.
    iters = []
    for name in os.listdir(save_dir):
        if name.startswith("iter_") and os.path.isdir(os.path.join(save_dir, name)):
            try:
                iters.append((int(name[len("iter_"):]), name))
            except ValueError:
                continue
    cur_iter = rollout_id
    # Delete every NON-permanent dir strictly older than the one just saved.
    # (Keeps: all permanents + the just-saved dir = rolling window of 1.)
    for it, name in iters:
        if it < cur_iter and not is_permanent(it):
            path = os.path.join(save_dir, name)
            try:
                shutil.rmtree(path)
                print(f"[retention] pruned non-permanent checkpoint {name}", flush=True)
                # Mirror deletion to S3. The bootstrap's background sync is additive
                # (no --delete), so without this the pruned dir would linger in S3.
                # Map local --save dir to its S3 location via the exported root pair.
                local_root = os.environ.get("CKPT_LOCAL_MODEL_ROOT")
                s3_root = os.environ.get("CKPT_S3_MODEL_ROOT")
                if local_root and s3_root and os.path.abspath(save_dir).startswith(os.path.abspath(local_root)):
                    rel = os.path.relpath(path, local_root)
                    s3_target = f"{s3_root.rstrip('/')}/{rel}"
                    import subprocess
                    subprocess.run(
                        ["aws", "s3", "rm", s3_target, "--recursive", "--only-show-errors"],
                        check=False,
                    )
                    print(f"[retention] mirrored S3 delete: {s3_target}", flush=True)
            except Exception as e:
                print(f"[retention] failed to prune {name}: {e}", flush=True)


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    def save(rollout_id):
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if actor_trains_this_step:
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)
            _prune_old_checkpoints(args, rollout_id)

        offload_train(actor_trains_this_step)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
