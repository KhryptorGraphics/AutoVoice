import json
from datetime import datetime, timezone
import uuid

# Load current checkpoints
with open('/home/kp/thordrive/autofusion/autovoice/data/app_state/profile_checkpoints.json', 'r') as f:
    checkpoints = json.load(f)

profile_id = "f16189d9-f6a1-4eac-813a-2f89e6a92f65"

# Deactivate current active checkpoint
for ckpt_id, ckpt in checkpoints[profile_id].items():
    if ckpt.get('is_active'):
        ckpt['is_active'] = False
        break

# Create new checkpoint for svc-fork epoch 100
new_ckpt_id = uuid.uuid4().hex
new_checkpoint = {
    'id': new_ckpt_id,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'epoch': 100,
    'model_version': 'svcfork_epoch100',
    'version': 'svcfork_epoch100',
    'active_model_type': 'svc_fork',
    'selected_adapter': None,
    'is_active': True,
    'profile_snapshot': {
        'has_trained_model': True,
        'loss_final': 0.0,
        'model_path': 'checkpoints/svcfork/G_100.pth',
        'model_version': 'svcfork_epoch100',
        'runtime_artifact_manifest_path': 'checkpoints/svcfork/config.json',
        'training_epochs': 100,
        'training_status': 'ready',
        'engine': 'so-vits-svc-fork',
        'speaker_id': 'connor',
        'config_path': 'checkpoints/svcfork/config.json',
    }
}

checkpoints[profile_id][new_ckpt_id] = new_checkpoint

# Write back
with open('/home/kp/thordrive/autofusion/autovoice/data/app_state/profile_checkpoints.json', 'w') as f:
    json.dump(checkpoints, f, indent=2)

print(f"Created checkpoint: {new_ckpt_id}")
print(f"Checkpoint: {json.dumps(new_checkpoint, indent=2)}")
