import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js";

const socket = io();

socket.on('connect', () => {
    console.log('Socket.IO connected');
});

socket.on('disconnect', () => {
    console.log('Socket.IO disconnected');
});

socket.on('status', (data) => {
    console.log('Status:', data);
});

// COMMENT 8 FIX: Add helper functions for song conversion WebSocket events

/**
 * Start song conversion stream
 * @param {Object} params - Conversion parameters
 * @param {string} params.conversion_id - Unique conversion ID
 * @param {string} params.song_data - Base64 encoded song audio
 * @param {string} params.target_profile_id - Target voice profile ID
 * @param {number} params.vocal_volume - Vocal volume (0-2)
 * @param {number} params.instrumental_volume - Instrumental volume (0-2)
 * @param {boolean} params.return_stems - Whether to return separated stems
 */
export function emitConvertSongStream(params) {
    socket.emit('convert_song_stream', params);
}

/**
 * Cancel ongoing conversion
 * @param {string} conversionId - Conversion ID to cancel
 */
export function emitCancelConversion(conversionId) {
    socket.emit('cancel_conversion', { conversion_id: conversionId });
}

/**
 * Get conversion status
 * @param {string} conversionId - Conversion ID to query
 */
export function emitGetConversionStatus(conversionId) {
    socket.emit('get_conversion_status', { conversion_id: conversionId });
}

/**
 * Register callback for conversion progress updates
 * @param {Function} callback - Callback function (data) => void
 * @returns {Function} Unsubscribe function
 */
export function onConversionProgress(callback) {
    socket.on('conversion_progress', callback);
    return () => socket.off('conversion_progress', callback);
}

/**
 * Register callback for conversion completion
 * @param {Function} callback - Callback function (data) => void
 * @returns {Function} Unsubscribe function
 */
export function onConversionComplete(callback) {
    socket.on('conversion_complete', callback);
    return () => socket.off('conversion_complete', callback);
}

/**
 * Register callback for conversion errors
 * @param {Function} callback - Callback function (data) => void
 * @returns {Function} Unsubscribe function
 */
export function onConversionError(callback) {
    socket.on('conversion_error', callback);
    return () => socket.off('conversion_error', callback);
}

/**
 * Register callback for conversion cancellation
 * @param {Function} callback - Callback function (data) => void
 * @returns {Function} Unsubscribe function
 */
export function onConversionCancelled(callback) {
    socket.on('conversion_cancelled', callback);
    return () => socket.off('conversion_cancelled', callback);
}

/**
 * Register callback for conversion status updates
 * @param {Function} callback - Callback function (data) => void
 * @returns {Function} Unsubscribe function
 */
export function onConversionStatus(callback) {
    socket.on('conversion_status', callback);
    return () => socket.off('conversion_status', callback);
}

/**
 * Reconnect WebSocket if disconnected
 */
export function reconnect() {
    if (!socket.connected) {
        socket.connect();
    }
}

/**
 * Check if socket is connected
 * @returns {boolean}
 */
export function isConnected() {
    return socket.connected;
}

export default socket;
