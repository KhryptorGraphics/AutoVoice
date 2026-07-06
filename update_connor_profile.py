import json

# Load Connor profile
with open('/home/kp/thordrive/autofusion/autovoice/data/voice_profiles/f16189d9-f6a1-4eac-813a-2f89e6a92f65.json', 'r') as f:
    profile = json.load(f)

# Update profile to use svc-fork epoch 100
profile['active_model_type'] = 'svc_fork'
profile['model_version'] = 'svcfork_epoch100'
profile['model_path'] = 'checkpoints/svcfork/G_100.pth'
profile['training_epochs'] = 100
profile['training_status'] = 'ready'
profile['engine'] = 'so-vits-svc-fork'
profile['speaker_id'] = 'connor'
profile['config_path'] = 'checkpoints/svcfork/config.json'

# Save back
with open('/home/kp/thordrive/autofusion/autovoice/data/voice_profiles/f16189d9-f6a1-4eac-813a-2f89e6a92f65.json', 'w') as f:
    json.dump(profile, f, indent=2)

print("Updated Connor profile for svc-fork epoch 100")
print(f"active_model_type: {profile['active_model_type']}")
print(f"model_version: {profile['model_version']}")
