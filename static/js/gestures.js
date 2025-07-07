/**
 * Gesture handling system for Virtual Literature Companion
 * 
 * This module manages:
 * - Gesture parsing and application
 * - Animation choreography
 * - Audio synchronization
 * - UI transformations
 */

class GestureManager {
    constructor() {
        this.activeGestures = new Map();
        this.gestureQueue = [];
        this.isProcessing = false;
        
        // Gesture type mappings
        this.gestureTypes = {
            LEAN_IN: 'lean-in',
            PULL_BACK: 'pull-back',
            TREMBLE: 'tremble',
            ILLUMINATE: 'illuminate',
            FRAGMENT: 'fragment',
            WHISPER: 'whisper',
            GRIP: 'grip',
            SHATTER: 'shatter',
            BREATHE: 'breathe',
            REACH: 'reach',
            DANCE: 'dance'
        };
        
        // Audio elements
        this.audioElements = {};
        this.initializeAudio();
        
        // Screen effect overlay
        this.createScreenEffectOverlay();
    }
    
    initializeAudio() {
        // Map gesture types to audio elements
        Object.entries(this.gestureTypes).forEach(([key, value]) => {
            const audioElement = document.getElementById(`audio-${value}`);
            if (audioElement) {
                this.audioElements[key] = audioElement;
                // Set volume for different gestures
                audioElement.volume = this.getGestureVolume(key);
            }
        });
    }
    
    getGestureVolume(gestureType) {
        // Customize volume for different gesture types
        const volumes = {
            WHISPER: 0.3,
            TREMBLE: 0.4,
            SHATTER: 0.8,
            BREATHE: 0.2,
            default: 0.5
        };
        return volumes[gestureType] || volumes.default;
    }
    
    createScreenEffectOverlay() {
        // Create overlay element for screen-wide effects
        const overlay = document.createElement('div');
        overlay.id = 'screen-effect-overlay';
        overlay.className = 'screen-effect-overlay';
        document.body.appendChild(overlay);
        this.screenOverlay = overlay;
    }
    
    /**
     * Apply gestures to a message element
     * @param {HTMLElement} messageElement - The message container
     * @param {Array} gestures - Array of gesture objects
     */
    async applyGestures(messageElement, gestures) {
        if (!gestures || gestures.length === 0) return;
        
        // Queue gestures for processing
        this.gestureQueue.push({ element: messageElement, gestures });
        
        // Process queue if not already processing
        if (!this.isProcessing) {
            await this.processGestureQueue();
        }
    }
    
    async processGestureQueue() {
        this.isProcessing = true;
        
        while (this.gestureQueue.length > 0) {
            const { element, gestures } = this.gestureQueue.shift();
            
            // Group gestures by timing
            const gestureGroups = this.groupGesturesByTiming(gestures);
            
            // Process each group
            for (const group of gestureGroups) {
                await this.processGestureGroup(element, group);
            }
        }
        
        this.isProcessing = false;
    }
    
    groupGesturesByTiming(gestures) {
        // Group gestures that should execute simultaneously
        const groups = [];
        let currentGroup = [];
        let lastTimestamp = null;
        
        gestures.forEach(gesture => {
            if (!lastTimestamp || Math.abs(gesture.timestamp - lastTimestamp) < 100) {
                currentGroup.push(gesture);
            } else {
                if (currentGroup.length > 0) {
                    groups.push(currentGroup);
                }
                currentGroup = [gesture];
            }
            lastTimestamp = gesture.timestamp;
        });
        
        if (currentGroup.length > 0) {
            groups.push(currentGroup);
        }
        
        return groups;
    }
    
    async processGestureGroup(element, gestures) {
        // Apply all gestures in the group simultaneously
        const promises = gestures.map(gesture => this.applyGesture(element, gesture));
        await Promise.all(promises);
    }
    
    async applyGesture(element, gesture) {
        const gestureClass = `gesture-${this.gestureTypes[gesture.type]}`;
        const gestureId = `${gesture.type}-${Date.now()}`;
        
        // Store active gesture
        this.activeGestures.set(gestureId, {
            element,
            gesture,
            startTime: Date.now()
        });
        
        // Apply gesture-specific logic
        switch (gesture.type) {
            case 'ILLUMINATE':
                this.applyIlluminate(element, gesture.parameters);
                break;
                
            case 'FRAGMENT':
                this.applyFragment(element, gesture.parameters);
                break;
                
            case 'SHATTER':
                this.applyShatter(element);
                break;
                
            case 'REACH':
                this.applyReach(element, gesture.parameters);
                break;
                
            case 'LEAN_IN':
            case 'PULL_BACK':
                this.applyContainerGesture(gesture.type);
                break;
                
            case 'BREATHE':
                this.applyBreathe();
                break;
                
            default:
                element.classList.add(gestureClass);
        }
        
        // Play audio if available
        this.playGestureAudio(gesture.type);
        
        // Schedule cleanup
        setTimeout(() => {
            this.cleanupGesture(element, gestureId, gestureClass);
        }, gesture.duration * 1000);
        
        // Return a promise that resolves when the gesture completes
        return new Promise(resolve => {
            setTimeout(resolve, gesture.duration * 1000);
        });
    }
    
    applyIlluminate(element, parameters) {
        if (!parameters || parameters.length === 0) return;
        
        // Find and highlight specified words
        parameters.forEach(word => {
            const regex = new RegExp(`\\b${word}\\b`, 'gi');
            element.innerHTML = element.innerHTML.replace(regex, match => {
                return `<span class="illuminate-word">${match}</span>`;
            });
        });
        
        element.classList.add('gesture-illuminate');
    }
    
    applyFragment(element, parameters) {
        if (!parameters || parameters.length === 0) return;
        
        // Fragment specified words
        parameters.forEach(word => {
            const regex = new RegExp(`\\b${word}\\b`, 'gi');
            element.innerHTML = element.innerHTML.replace(regex, match => {
                const letters = match.split('').map(letter => 
                    `<span class="fragment-letter">${letter}</span>`
                ).join('');
                return `<span class="fragment-word">${letters}</span>`;
            });
        });
        
        element.classList.add('gesture-fragment');
    }
    
    applyShatter(element) {
        // Apply shatter effect to the element
        element.classList.add('gesture-shatter');
        
        // Add screen-wide effect
        this.screenOverlay.classList.add('screen-effect-shatter');
        setTimeout(() => {
            this.screenOverlay.classList.remove('screen-effect-shatter');
        }, 3000);
    }
    
    applyReach(element, parameters) {
        // Create visual connections between words/concepts
        element.classList.add('gesture-reach');
        
        if (parameters && parameters.length >= 2) {
            // Create connection lines between specified words
            this.createWordConnections(element, parameters[0], parameters[1]);
        }
    }
    
    applyContainerGesture(gestureType) {
        const appContainer = document.getElementById('app');
        const gestureClass = `gesture-${this.gestureTypes[gestureType]}`;
        
        appContainer.classList.add(gestureClass);
        
        // Remove after duration
        setTimeout(() => {
            appContainer.classList.remove(gestureClass);
        }, gestureType === 'LEAN_IN' ? 2000 : 2500);
    }
    
    applyBreathe() {
        const appContainer = document.getElementById('app');
        appContainer.classList.add('gesture-breathe');
        
        // Breathe is continuous, so we don't auto-remove
        // It should be removed when a new gesture is applied
    }
    
    createWordConnections(element, word1, word2) {
        // Find positions of words
        const spans = element.querySelectorAll('span');
        let pos1 = null, pos2 = null;
        
        spans.forEach(span => {
            if (span.textContent.includes(word1) && !pos1) {
                pos1 = span.getBoundingClientRect();
            }
            if (span.textContent.includes(word2) && !pos2) {
                pos2 = span.getBoundingClientRect();
            }
        });
        
        if (pos1 && pos2) {
            // Create connection line
            const connection = document.createElement('div');
            connection.className = 'reach-connection';
            
            // Calculate position and angle
            const dx = pos2.left - pos1.left;
            const dy = pos2.top - pos1.top;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const angle = Math.atan2(dy, dx) * 180 / Math.PI;
            
            connection.style.width = `${distance}px`;
            connection.style.transform = `rotate(${angle}deg)`;
            connection.style.left = `${pos1.left}px`;
            connection.style.top = `${pos1.top + pos1.height / 2}px`;
            
            element.appendChild(connection);
        }
    }
    
    playGestureAudio(gestureType) {
        const audio = this.audioElements[gestureType];
        if (audio) {
            // Clone and play to allow overlapping sounds
            const audioClone = audio.cloneNode();
            audioClone.volume = audio.volume;
            audioClone.play().catch(err => {
                console.warn('Audio playback failed:', err);
            });
        }
    }
    
    cleanupGesture(element, gestureId, gestureClass) {
        // Remove gesture from active list
        this.activeGestures.delete(gestureId);
        
        // Remove gesture class
        element.classList.remove(gestureClass);
        
        // Clean up any gesture-specific elements
        const illuminateWords = element.querySelectorAll('.illuminate-word');
        const fragmentWords = element.querySelectorAll('.fragment-word');
        const connections = element.querySelectorAll('.reach-connection');
        
        // Gradually fade out effects
        [...illuminateWords, ...fragmentWords].forEach(el => {
            el.style.transition = 'all 0.5s ease-out';
            el.style.opacity = '0.5';
        });
        
        connections.forEach(el => el.remove());
    }
    
    /**
     * Clear all active gestures
     */
    clearAllGestures() {
        this.activeGestures.forEach((data, gestureId) => {
            const gestureClass = `gesture-${this.gestureTypes[data.gesture.type]}`;
            this.cleanupGesture(data.element, gestureId, gestureClass);
        });
        
        // Clear container gestures
        const appContainer = document.getElementById('app');
        Object.values(this.gestureTypes).forEach(type => {
            appContainer.classList.remove(`gesture-${type}`);
        });
    }
    
    /**
     * Get currently active gestures
     */
    getActiveGestures() {
        return Array.from(this.activeGestures.values());
    }
}

// Create global instance
window.gestureManager = new GestureManager();