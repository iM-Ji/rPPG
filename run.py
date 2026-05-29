import os
import torch
import multiprocessing as mp
from optparse import OptionParser

torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"]        = "2"
os.environ["MKL_NUM_THREADS"]        = "2"
os.environ["OPENBLAS_NUM_THREADS"]   = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"]    = "2"
os.environ["QT_LOGGING_RULES"]       = "qt.qpa.fonts=false"

from capture_frames import CaptureFrames
from process_mask import RunPOS


def get_args():
    parser = OptionParser()
    parser.add_option('-s', '--source',     dest='source',    default="0")
    parser.add_option('-b', '--batch-size', dest='batchsize', type='int', default=10)
    parser.add_option('-f', '--frame-rate', dest='framerate', type='int', default=15)
    parser.add_option('--no-plot',          dest='plot',      action='store_false', default=True)
    parser.add_option('--model',            dest='model',     default=None)
    (options, _) = parser.parse_args()
    return options


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args   = get_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    if args.model is None:
        base = os.path.dirname(os.path.abspath(__file__))
        for candidate in [
            os.path.join(base, "models", "best_model.pth"),
            os.path.join(base, "best_model.pth"),
            os.path.join(base, "models", "best_model_cpu_optimized.pt"),
            os.path.join(base, "best_model_cpu_optimized.pt"),
            os.path.join(base, "models", "nasir6_optimized.onnx"),
        ]:
            if os.path.exists(candidate):
                args.model = candidate
                break

    if args.model is None:
        print("❌ No model file found. Use --model /path/to/model")
        exit(1)

    mask_send, mask_recv = mp.Pipe()
    res_send,  res_recv  = mp.Pipe()

    capture_engine = CaptureFrames(bs=args.batchsize, source=source, show_mask=True)
    math_engine    = RunPOS(
        batch_size=args.batchsize,
        framerate=args.framerate,
        model_path=args.model,
        plot=args.plot
    )

    p_math = mp.Process(target=math_engine, args=(mask_recv, res_send))
    p_math.daemon = True
    p_math.start()

    print("\n" + "=" * 45)
    print("  rPPG SYSTEM — CPU OPTIMISED BUILD")
    print("=" * 45)
    print(f"  Camera source : {source}")
    print(f"  Camera FPS    : {args.framerate}")
    print(f"  Batch size    : {args.batchsize}")
    print(f"  Model         : {os.path.basename(args.model)}")
    print(f"  Face detect   : Haar cascade (CPU only)")
    print(f"  Threads/proc  : 2")
    print("  Press Q or ESC to quit.")
    print("  Stay still ~17s to fill the signal buffer.")
    print("=" * 45 + "\n")

    try:
        capture_engine(mask_send, res_recv, source)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        p_math.terminate()
        p_math.join(timeout=3)
        print("Shutdown complete.")