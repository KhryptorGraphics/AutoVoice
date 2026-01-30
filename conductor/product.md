# Product Definition

## Project Name

AutoVoice

## Description

GPU-accelerated singing voice conversion and TTS system that converts songs to a target voice while preserving pitch and timing, using the So-VITS-SVC architecture.

## Problem Statement

Existing voice conversion tools can't preserve pitch and timing during voice swap. Current solutions either require expensive cloud APIs with latency, or produce output that loses the musical qualities of the original performance.

## Target Users

Music producers and audio engineers who need high-quality vocal transformation for production work.

## Key Goals

1. **Real-time inference on edge hardware (Jetson Thor)** - Sub-100ms latency voice conversion running entirely on-device
2. **Production-quality vocal output preserving musicality** - Output indistinguishable from the original singer's performance quality
3. **Simple API for integration into DAWs and production tools** - Easy-to-use REST/WebSocket API for third-party integration
4. **End-to-end voice cloning pipeline** - Input a song, extract the vocal track, train a model on the user's voice from training data, replace the artist's voice with the user's voice, giving the user the singing ability of the artist in the song

## Core Workflow

1. User provides a song (input audio)
2. System extracts/separates the vocal track
3. User provides training data of their own voice
4. System trains a voice model on the user's voice
5. System converts the extracted vocal to the user's voice
6. Output: the song with the user's voice, preserving the original artist's pitch and timing
