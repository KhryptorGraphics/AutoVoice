# AutoVoice Browser Automation Testing - Resume Point

## Session State (2026-01-30 10:30 AM CST) - PRE-COMPACTION SAVE

### COMPLETED
- [x] Phase 1: Basic Navigation Testing
- [x] Phase 2: System Status Page Testing
- [x] Phase 3: Voice Profiles CRUD Testing
- [x] Progressive Training Architecture implemented
- [x] Downloaded 14 additional training songs (8 Conor, 6 William)

### IN PROGRESS - Phase 4: Training Flow Testing

**IMMEDIATE NEXT STEPS:**
1. Train initial models on Pillowtalk (same song, different styles)
2. Show LIVE training progress to user during training
3. Voice conversion: Connor→William instrumental, William→Connor instrumental
4. Quality assessment: built-in metrics + user listening test
5. Collaborate on quality - iterate until acceptable
6. Update web UI for progressive training uploads

### DOWNLOADED TRAINING SONGS
**Conor Maynard** (`data/youtube_downloads/conor_maynard/`):
- stitches.wav, pill_in_ibiza.wav, 7_years.wav, hello.wav
- faded.wav, runaway_man.wav, only_one.wav, slow_dancing.wav

**William Singe** (`data/youtube_downloads/william_singe/`):
- say_my_name.wav, hotline_bling.wav, lemonade.wav
- cry_me_a_river.wav, no_scrubs.wav, bad_and_boujee.wav

### PILLOWTALK TEST FILES
- `tests/quality_samples/conor_maynard_pillowtalk.wav`
- `tests/quality_samples/william_singe_pillowtalk.wav`

### EXISTING PROFILES
- William Singe: `7da05140-1303-40c6-95d9-5b6e2c3624df`
- Conor Maynard: `9679a6ec-e6e2-43c4-b64e-1f004fed34f9`

### BUGS FIXED
1. **Profile names showing as UUIDs** - Fixed in api.ts, voice_cloner.py, api.py
2. **Automatic vocal separation** - Demucs HTDemucs, stores vocals_extracted
3. **Progressive sample storage** - TrainingSample class in voice_profiles.py

### WEB UI UPDATES NEEDED
1. Add "Upload Song" button on profile detail page
2. Show training sample count and duration
3. Live training progress display (WebSocket events ready)
4. Quality metrics display after training

### KEY PIPELINES
- SOTAConversionPipeline: MelBandRoFormer→ContentVec→RMVPE→CoMoSVC→BigVGAN
- FineTuningPipeline: LoRA adapter training with EWC
- TrainingJobManager: job queue with WebSocket events

### NEXT TASK
Full voice conversion pipeline test:
1. Create proper Conor Maynard profile (with vocal separation)
2. Train models on both artists via web interface
3. Voice swap: Connor → William instrumental, William → Connor instrumental

### KEY FILES
- Test audio: `tests/quality_samples/conor_maynard_pillowtalk.wav`
- Test audio: `test_audio/William Singe - Pillowtalk (Cover).mp3`
- Browser tool: `agent-browser` CLI

### PROFILES CREATED
| Name | ID | Vocals Extracted |
|------|-----|-----------------|
| Conor Maynard | e3138fa2-b9af-474b-8749-c24a931fc307 | NO (pre-fix) |
| William Singe | 7da05140-1303-40c6-95d9-5b6e2c3624df | YES |

### REMAINING PHASES
- [ ] Phase 4: Training Flow Testing
- [ ] Phase 5: Karaoke Page Testing
- [ ] Phase 6: Convert Page Testing
- [ ] Phase 7: Error Handling Testing
