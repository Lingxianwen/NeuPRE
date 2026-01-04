"""
Debug script to understand boundary prediction issues
"""

import logging
logging.basicConfig(level=logging.INFO)

from modules.format_learner import InformationBottleneckFormatLearner
from utils.dynpre_loader import DynPREGroundTruthLoader
from experiments.experiment2_segmentation import simulate_dynpre_segmentation

# Load DynPRE ground truth
loader = DynPREGroundTruthLoader(dynpre_output_dir='../../DynPRE/examples')
messages, ground_truth = loader.load_ground_truth('modbus')

# Use first 20 messages for training
train_messages = messages[:20]
test_messages = messages[20:23]
test_ground_truth = ground_truth[20:23]

print("="*80)
print("TRAINING NeuPRE on 20 Modbus messages...")
print("="*80)

# Train NeuPRE
learner = InformationBottleneckFormatLearner(d_model=128, nhead=4, num_layers=3, beta=0.1)
learner.train(train_messages, None, epochs=30, batch_size=16)

print("\n" + "="*80)
print("TESTING on 3 unseen messages:")
print("="*80)

for i in range(3):
    msg = test_messages[i]
    gt = test_ground_truth[i]

    # Get predictions
    neupre_pred = learner.extract_boundaries(msg, threshold=0.5)
    dynpre_pred = simulate_dynpre_segmentation([msg])[0]

    print(f"\nMessage {i+1}: {msg.hex()}")
    print(f"  Length: {len(msg)} bytes")
    print(f"  Ground truth:  {gt}  ({len(gt)-1} fields)")
    print(f"  NeuPRE pred:   {neupre_pred}  ({len(neupre_pred)-1} fields)")
    print(f"  DynPRE pred:   {dynpre_pred}  ({len(dynpre_pred)-1} fields)")

    # Compute overlap
    gt_set = set(gt)
    neupre_set = set(neupre_pred)
    dynpre_set = set(dynpre_pred)

    neupre_tp = len(gt_set & neupre_set)
    neupre_fp = len(neupre_set - gt_set)
    neupre_fn = len(gt_set - neupre_set)

    dynpre_tp = len(gt_set & dynpre_set)
    dynpre_fp = len(dynpre_set - gt_set)
    dynpre_fn = len(gt_set - dynpre_set)

    print(f"\n  NeuPRE: TP={neupre_tp}, FP={neupre_fp}, FN={neupre_fn}")
    print(f"  DynPRE: TP={dynpre_tp}, FP={dynpre_fp}, FN={dynpre_fn}")

    if neupre_tp + neupre_fp > 0:
        neupre_precision = neupre_tp / (neupre_tp + neupre_fp)
    else:
        neupre_precision = 0

    if neupre_tp + neupre_fn > 0:
        neupre_recall = neupre_tp / (neupre_tp + neupre_fn)
    else:
        neupre_recall = 0

    if dynpre_tp + dynpre_fp > 0:
        dynpre_precision = dynpre_tp / (dynpre_tp + dynpre_fp)
    else:
        dynpre_precision = 0

    if dynpre_tp + dynpre_fn > 0:
        dynpre_recall = dynpre_tp / (dynpre_tp + dynpre_fn)
    else:
        dynpre_recall = 0

    print(f"\n  NeuPRE: Precision={neupre_precision:.4f}, Recall={neupre_recall:.4f}")
    print(f"  DynPRE: Precision={dynpre_precision:.4f}, Recall={dynpre_recall:.4f}")

print("\n" + "="*80)
print("TRYING DIFFERENT THRESHOLDS:")
print("="*80)

test_msg = test_messages[0]
test_gt = test_ground_truth[0]

for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    pred = learner.extract_boundaries(test_msg, threshold=threshold)
    tp = len(set(test_gt) & set(pred))
    fp = len(set(pred) - set(test_gt))
    fn = len(set(test_gt) - set(pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Threshold={threshold:.1f}: {len(pred)-1} fields, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
    print(f"    Predicted: {pred}")