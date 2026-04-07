# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.utils.experiment import (
    create_run_layout,
    finalize_stage_manifest,
    should_write_manifest,
    start_stage_manifest,
)
from wan.utils.animate_contract import build_preprocess_metadata, write_preprocess_metadata


def _parse_args():
    parser = argparse.ArgumentParser(
        description="The preprocessing pipeline for Wan-animate."
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="The path to the preprocessing model's checkpoint directory. ")

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="The path to the driving video.")
    parser.add_argument(
        "--refer_path",
        type=str,
        default=None,
        help="The path to the refererence image.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="The path to save the processed results.")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Logical name for the current preprocessing run. If set, a standard run directory is created under ./runs unless --run_dir is also provided."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directory for the current experiment run. When set, the preprocess stage writes outputs and manifest metadata into the standardized run layout."
    )
    parser.add_argument(
        "--save_manifest",
        action="store_true",
        default=False,
        help="Write a stage manifest for this invocation. Automatically enabled when --run_name or --run_dir is provided."
    )
    
    parser.add_argument(
        "--resolution_area",
        type=int,
        nargs=2,
        default=[1280, 720],
        help="The target resolution for processing, specified as [width, height]. To handle different aspect ratios, the video is resized to have a total area equivalent to width * height, while preserving the original aspect ratio."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="The target FPS for processing the driving video. Set to -1 to use the video's original FPS."
    )

    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Whether to use replacement mode.")
    parser.add_argument(
        "--retarget_flag",
        action="store_true",
        default=False,
        help="Whether to use pose retargeting. Currently only supported in animation mode")
    parser.add_argument(
        "--use_flux",
        action="store_true",
        default=False,
        help="Whether to use image editing in pose retargeting. Recommended if the character in the reference image or the first frame of the driving video is not in a standard, front-facing pose")
    
    # Parameters for the mask strategy in replacement mode. These control the mask's size and shape. Refer to https://arxiv.org/pdf/2502.06145
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations for mask dilation."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=7,
        help="Number of kernel size for mask dilation."
    )
    parser.add_argument(
        "--w_len",
        type=int,
        default=1,
        help="The number of subdivisions for the grid along the 'w' dimension. A higher value results in a more detailed contour. A value of 1 means no subdivision is performed."
    )
    parser.add_argument(
        "--h_len",
        type=int,
        default=1,
        help="The number of subdivisions for the grid along the 'h' dimension. A higher value results in a more detailed contour. A value of 1 means no subdivision is performed."
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = _parse_args()
    args_dict = vars(args)
    print(args_dict)

    assert len(args.resolution_area) == 2, "resolution_area should be a list of two integers [width, height]"
    assert not args.use_flux or args.retarget_flag, "Image editing with FLUX can only be used when pose retargeting is enabled."
    assert args.ckpt_path is not None, "Please provide --ckpt_path."
    assert Path(args.ckpt_path).exists(), f"Checkpoint path does not exist: {args.ckpt_path}"
    assert args.video_path is not None, "Please provide --video_path."
    assert Path(args.video_path).exists(), f"Video path does not exist: {args.video_path}"
    assert args.refer_path is not None, "Please provide --refer_path."
    assert Path(args.refer_path).exists(), f"Reference image path does not exist: {args.refer_path}"
    run_layout = None
    manifest_token = None
    if should_write_manifest(args):
        if args.run_name is None and args.run_dir is None:
            base = Path(args.save_path).name if args.save_path is not None else "preprocess"
            args.run_name = f"{base}_{Path(args.video_path).stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        run_layout = create_run_layout(run_name=args.run_name, run_dir=args.run_dir, repo_root=REPO_ROOT)
        if args.save_path is None:
            args.save_path = str(run_layout["preprocess_dir"].resolve())
    assert args.save_path is not None, "Please provide --save_path or enable run tracking with --run_name/--run_dir."

    pose2d_checkpoint_path = os.path.join(args.ckpt_path, 'pose2d/vitpose_h_wholebody.onnx')
    det_checkpoint_path = os.path.join(args.ckpt_path, 'det/yolov10m.onnx')

    sam2_checkpoint_path = os.path.join(args.ckpt_path, 'sam2/sam2_hiera_large.pt') if args.replace_flag else None
    flux_kontext_path = os.path.join(args.ckpt_path, 'FLUX.1-Kontext-dev') if args.use_flux else None
    from process_pipepline import ProcessPipeline
    process_pipeline = ProcessPipeline(det_checkpoint_path=det_checkpoint_path, pose2d_checkpoint_path=pose2d_checkpoint_path, sam_checkpoint_path=sam2_checkpoint_path, flux_kontext_path=flux_kontext_path)
    os.makedirs(args.save_path, exist_ok=True)
    if run_layout is not None:
        manifest_token = start_stage_manifest(
            run_layout,
            "preprocess",
            args,
            inputs={
                "video_path": args.video_path,
                "refer_path": args.refer_path,
                "checkpoint_path": args.ckpt_path,
            },
        )
    try:
        pipeline_outputs = process_pipeline(video_path=args.video_path,
                                            refer_image_path=args.refer_path,
                                            output_path=args.save_path,
                                            resolution_area=args.resolution_area,
                                            fps=args.fps,
                                            iterations=args.iterations,
                                            k=args.k,
                                            w_len=args.w_len,
                                            h_len=args.h_len,
                                            retarget_flag=args.retarget_flag,
                                            use_flux=args.use_flux,
                                            replace_flag=args.replace_flag)
        metadata = build_preprocess_metadata(
            video_path=args.video_path,
            refer_image_path=args.refer_path,
            output_path=args.save_path,
            replace_flag=args.replace_flag,
            retarget_flag=args.retarget_flag,
            use_flux=args.use_flux,
            resolution_area=args.resolution_area,
            fps_request=args.fps,
            fps_output=pipeline_outputs["fps"],
            frame_count=pipeline_outputs["frame_count"],
            height=pipeline_outputs["height"],
            width=pipeline_outputs["width"],
            iterations=args.iterations,
            k=args.k,
            w_len=args.w_len,
            h_len=args.h_len,
            reference_height=pipeline_outputs["reference_height"],
            reference_width=pipeline_outputs["reference_width"],
        )
        metadata_path = write_preprocess_metadata(args.save_path, metadata)
        if manifest_token is not None:
            output_dir = Path(args.save_path).resolve()
            stage_outputs = {
                "save_path": str(output_dir),
                "generated_files": sorted(str(path.resolve()) for path in output_dir.iterdir()),
                "metadata_path": str(metadata_path.resolve()),
            }
            for name in ["src_pose.mp4", "src_face.mp4", "src_bg.mp4", "src_mask.mp4", "src_ref.png", "metadata.json"]:
                path = output_dir / name
                if path.exists():
                    stage_outputs[path.stem] = str(path)
            finalize_stage_manifest(
                run_layout,
                manifest_token,
                status="completed",
                outputs=stage_outputs,
            )
    except Exception as exc:
        if manifest_token is not None:
            finalize_stage_manifest(
                run_layout,
                manifest_token,
                status="failed",
                outputs={"save_path": str(Path(args.save_path).resolve())},
                error=str(exc),
            )
        raise
