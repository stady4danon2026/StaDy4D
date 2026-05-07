  Files:                                                                                                                                                            
  - sigma/evaluation/__init__.py — package init                                                                                                                   
  - sigma/evaluation/__main__.py — allows python -m sigma.evaluation                                                                                                
  - sigma/evaluation/metrics.py — all metric functions (pure numpy, no heavy deps)                                                                                  
  - sigma/evaluation/data_loader.py — loads GT/prediction directories (supports CARLA layout with static//dynamic/ subfolders)                                      
  - sigma/evaluation/evaluator.py — main Evaluator class orchestrating everything                                                                                   
  - sigma/evaluation/evaluate.py — CLI entry point                                                                                                                  

  Metrics implemented:
  ┌─────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │  Category   │                                                         Metrics                                                          │
  ├─────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Camera Pose │ Per-frame trans/rot error, ATE (RMSE with Procrustes alignment), RPE (delta=1), accuracy thresholds (acc@0.1m,1deg etc.) │
  ├─────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Depth       │ AbsRel, SqRel, RMSE, RMSE_log, SI_log, delta<1.25/1.25²/1.25³, with median or least-squares scale alignment              │
  ├─────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Point Cloud │ Chamfer distance (bidirectional), F-score with precision/recall                                                          │
  ├─────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Intrinsics  │ Relative errors on fx, fy, cx, cy                                                                                        │
  └─────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
  Usage:
  # All metrics
  python -m sigma.evaluation.evaluate \
      --gt data/carla/output/video_00 \
      --pred outputs/cc49 \
      --output eval_results.json

  # Pose-only
  python -m sigma.evaluation.evaluate \
      --gt data/carla/output/video_00 \
      --pred outputs/cc49 \
      --metrics pose

  # Custom depth alignment
  python -m sigma.evaluation.evaluate \
      --gt data/carla/output/video_00 \
      --pred outputs/cc49 \
      --depth-align lsq --max-depth 100