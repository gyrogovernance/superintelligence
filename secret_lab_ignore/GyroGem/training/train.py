# Two-Stage Training Pipeline
# [Authority:Indirect] + [Agency:Indirect]


def train_full_pipeline():
    """Run the complete two-stage training pipeline."""
    print("=" * 60)
    print("GyroGem Training Pipeline")
    print("=" * 60)

    # Stage 1: Domain absorption
    from .stage1_absorb import absorb_domain
    stage1_path = absorb_domain()

    # Stage 2: Task application
    from .stage2_classify import apply_task_finetuning
    final_path = apply_task_finetuning(stage1_path)

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print(f"Final model: {final_path}")
    print("=" * 60)

    return final_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        from .stage2_classify import apply_task_finetuning
        apply_task_finetuning(sys.argv[1])
    else:
        train_full_pipeline()