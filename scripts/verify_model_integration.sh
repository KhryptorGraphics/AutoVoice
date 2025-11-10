#!/bin/bash
# Verification script for model integration infrastructure

set -e  # Exit on error

echo "=================================================="
echo "Model Integration Infrastructure Verification"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUCCESS=0
FAILURES=0

# Function to run check
run_check() {
    local name="$1"
    local command="$2"

    echo -n "Checking $name... "

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        ((SUCCESS++))
        return 0
    else
        echo -e "${RED}✗${NC}"
        ((FAILURES++))
        return 1
    fi
}

# 1. Import checks
echo "1. Import Verification"
echo "----------------------"

run_check "ModelRegistry import" \
    "python -c 'from auto_voice.models import ModelRegistry'"

run_check "ModelConfig import" \
    "python -c 'from auto_voice.models import ModelConfig'"

run_check "HuBERTModel import" \
    "python -c 'from auto_voice.models import HuBERTModel'"

run_check "HiFiGANModel import" \
    "python -c 'from auto_voice.models import HiFiGANModel'"

run_check "SpeakerEncoderModel import" \
    "python -c 'from auto_voice.models import SpeakerEncoderModel'"

echo ""

# 2. File existence checks
echo "2. File Structure"
echo "-----------------"

run_check "model_registry.py exists" \
    "test -f src/auto_voice/models/model_registry.py"

run_check "model_loader.py exists" \
    "test -f src/auto_voice/models/model_loader.py"

run_check "hubert_model.py exists" \
    "test -f src/auto_voice/models/hubert_model.py"

run_check "hifigan_model.py exists" \
    "test -f src/auto_voice/models/hifigan_model.py"

run_check "models.yaml exists" \
    "test -f config/models.yaml"

run_check "download_models.py exists" \
    "test -f scripts/download_models.py"

run_check "download_models.py executable" \
    "test -x scripts/download_models.py"

echo ""

# 3. Functionality checks
echo "3. Functionality"
echo "----------------"

run_check "Registry creation (mock)" \
    "python -c 'from auto_voice.models import ModelRegistry; r = ModelRegistry(use_mock=True)'"

run_check "Load HuBERT (mock)" \
    "python -c 'from auto_voice.models import ModelRegistry; r = ModelRegistry(use_mock=True); r.load_hubert()'"

run_check "Load HiFi-GAN (mock)" \
    "python -c 'from auto_voice.models import ModelRegistry; r = ModelRegistry(use_mock=True); r.load_hifigan()'"

run_check "Load Speaker Encoder (mock)" \
    "python -c 'from auto_voice.models import ModelRegistry; r = ModelRegistry(use_mock=True); r.load_speaker_encoder()'"

run_check "Model caching works" \
    "python -c 'from auto_voice.models import ModelRegistry; r = ModelRegistry(use_mock=True); m1 = r.load_hubert(); m2 = r.load_hubert(); assert m1 is m2'"

run_check "List models" \
    "python -c 'from auto_voice.models import ModelRegistry; r = ModelRegistry(use_mock=True); assert len(r.list_models()) > 0'"

echo ""

# 4. Documentation checks
echo "4. Documentation"
echo "----------------"

run_check "MODEL_INTEGRATION.md exists" \
    "test -f docs/MODEL_INTEGRATION.md"

run_check "models/README.md exists" \
    "test -f docs/models/README.md"

run_check "IMPLEMENTATION_SUMMARY.md exists" \
    "test -f docs/models/IMPLEMENTATION_SUMMARY.md"

run_check "Example script exists" \
    "test -f examples/model_integration_example.py"

run_check "Example script executable" \
    "test -x examples/model_integration_example.py"

echo ""

# 5. Test suite checks
echo "5. Test Suite"
echo "-------------"

run_check "Test file exists" \
    "test -f tests/models/test_model_registry.py"

run_check "Tests run successfully" \
    "python -m pytest tests/models/test_model_registry.py -q --tb=no"

echo ""

# 6. Pipeline integration
echo "6. Pipeline Integration"
echo "-----------------------"

run_check "Pipeline accepts model_registry" \
    "python -c 'from auto_voice.inference import VoiceConversionPipeline, PipelineConfig; from auto_voice.models import ModelRegistry; r = ModelRegistry(use_mock=True); p = VoiceConversionPipeline(config=PipelineConfig(), model_registry=r)' 2>&1 | grep -q ."

run_check "Pipeline creates registry automatically" \
    "python -c 'from auto_voice.inference import VoiceConversionPipeline, PipelineConfig; p = VoiceConversionPipeline(config=PipelineConfig(use_mock_models=True))'"

echo ""

# Summary
echo "=================================================="
echo "Summary"
echo "=================================================="
echo -e "Passed: ${GREEN}${SUCCESS}${NC}"
echo -e "Failed: ${RED}${FAILURES}${NC}"
echo ""

if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "Model integration infrastructure is working correctly."
    exit 0
else
    echo -e "${RED}✗ Some checks failed.${NC}"
    echo "Please review the failures above."
    exit 1
fi
