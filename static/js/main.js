/**
 * Main application logic for Virtual Literature Companion
 * 
 * This module handles:
 * - WebSocket communication
 * - UI state management
 * - Message handling
 * - Session management
 */

class LiteraryCompanion {
    constructor() {
        this.sessionId = null;
        this.websocket = null;
        this.bookTitle = null;
        this.bookAuthor = null;
        this.currentLocation = null;
        this.tableOfContents = null;
        
        // UI elements
        this.setupScreen = document.getElementById('setup-screen');
        this.chatInterface = document.getElementById('chat-interface');
        this.messagesContainer = document.getElementById('messages');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.typingIndicator = document.getElementById('typing-indicator');
        
        // Initialize
        this.initializeEventListeners();
        this.adjustTextareaHeight();
    }
    
    initializeEventListeners() {
        // Setup screen
        document.getElementById('start-session').addEventListener('click', () => this.startSession());
        document.getElementById('book-file').addEventListener('change', (e) => this.handleFileUpload(e));
        
        // Chat interface
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => this.adjustTextareaHeight());
        
        // Handle page visibility for WebSocket reconnection
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.websocket?.readyState === WebSocket.CLOSED) {
                this.reconnectWebSocket();
            }
        });
    }
    
    adjustTextareaHeight() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 150) + 'px';
    }
    
    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            this.bookText = e.target.result;
            
            // Update UI to show file loaded
            const uploadLabel = document.querySelector('.file-upload-label');
            uploadLabel.innerHTML = `
                <span class="upload-icon">✓</span>
                <span class="upload-text">${file.name} loaded</span>
            `;
        };
        
        reader.readAsText(file);
    }
    
    async startSession() {
        const title = document.getElementById('book-title').value.trim();
        const author = document.getElementById('book-author').value.trim();
        
        if (!title || !author || !this.bookText) {
            alert('Please provide book title, author, and upload the text file.');
            return;
        }
        
        // Show loading state
        const startButton = document.getElementById('start-session');
        startButton.disabled = true;
        startButton.textContent = 'Initializing...';
        
        try {
            // Create session
            const response = await fetch('/api/sessions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title: title,
                    author: author,
                    text: this.bookText
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to create session');
            }
            
            const data = await response.json();
            
            // Store session data
            this.sessionId = data.session_id;
            this.bookTitle = data.book_title;
            this.bookAuthor = data.author;
            this.tableOfContents = data.table_of_contents;
            
            // Update UI
            document.getElementById('current-book-title').textContent = this.bookTitle;
            document.getElementById('current-book-author').textContent = `by ${this.bookAuthor}`;
            
            // Display table of contents
            if (this.tableOfContents) {
                this.displayTableOfContents();
            }
            
            // Connect WebSocket
            await this.connectWebSocket();
            
            // Transition to chat interface
            this.setupScreen.classList.remove('active');
            setTimeout(() => {
                this.chatInterface.classList.add('active');
                this.messageInput.focus();
            }, 300);
            
        } catch (error) {
            console.error('Error starting session:', error);
            alert('Failed to start session. Please try again.');
            startButton.disabled = false;
            startButton.textContent = 'Begin Reading Session';
        }
    }
    
    displayTableOfContents() {
        const tocContent = document.getElementById('toc-content');
        tocContent.innerHTML = '';
        
        if (!this.tableOfContents?.chapters) return;
        
        this.tableOfContents.chapters.forEach(chapter => {
            const chapterElement = document.createElement('div');
            chapterElement.className = 'toc-chapter';
            chapterElement.innerHTML = `
                <strong>Chapter ${chapter.number}:</strong> ${chapter.title}
                <span class="chapter-pages">(pp. ${chapter.start_page}-${chapter.end_page})</span>
            `;
            
            chapterElement.addEventListener('click', () => {
                this.insertProgressUpdate(chapter.number, chapter.start_page);
            });
            
            tocContent.appendChild(chapterElement);
        });
    }
    
    insertProgressUpdate(chapterNumber, pageNumber) {
        this.messageInput.value = `I've read up to Chapter ${chapterNumber}, page ${pageNumber}`;
        this.messageInput.focus();
        this.adjustTextareaHeight();
    }
    
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            this.websocket = new WebSocket(`ws://localhost:8000/ws/${this.sessionId}`);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                resolve();
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                // Attempt reconnection after a delay
                setTimeout(() => this.reconnectWebSocket(), 3000);
            };
        });
    }
    
    async reconnectWebSocket() {
        if (this.sessionId && (!this.websocket || this.websocket.readyState === WebSocket.CLOSED)) {
            try {
                await this.connectWebSocket();
            } catch (error) {
                console.error('Failed to reconnect:', error);
            }
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'message':
                this.displayMessage(data.content, 'assistant', data.gestures, data.citations);
                break;
                
            case 'system':
                this.displaySystemMessage(data.message);
                break;
                
            case 'typing':
                this.toggleTypingIndicator(data.status === 'start');
                break;
                
            case 'error':
                this.displayErrorMessage(data.message);
                break;
                
            case 'pong':
                // Keepalive response
                break;
        }
    }
    
    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        // Display user message
        this.displayMessage(message, 'user');
        
        // Send via WebSocket
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'message',
                content: message
            }));
        }
        
        // Clear input
        this.messageInput.value = '';
        this.adjustTextareaHeight();
    }
    
    displayMessage(content, role, gestures = [], citations = []) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${role}`;
        
        const contentElement = document.createElement('div');
        contentElement.className = 'message-content';
        contentElement.innerHTML = this.formatMessageContent(content, citations);
        
        const timeElement = document.createElement('div');
        timeElement.className = 'message-time';
        timeElement.textContent = new Date().toLocaleTimeString();
        
        messageElement.appendChild(contentElement);
        messageElement.appendChild(timeElement);
        
        this.messagesContainer.appendChild(messageElement);
        
        // Apply gestures if this is an assistant message
        if (role === 'assistant' && gestures.length > 0) {
            window.gestureManager.applyGestures(contentElement, gestures);
        }
        
        // Scroll to bottom
        this.scrollToBottom();
    }
    
    formatMessageContent(content, citations) {
        // Format paragraphs
        let formatted = content.split('\n\n').map(p => `<p>${p}</p>`).join('');
        
        // Format citations
        if (citations && citations.length > 0) {
            citations.forEach((citation, index) => {
                const citationMark = `<sup class="citation-mark" data-citation="${index}">[${index + 1}]</sup>`;
                // Insert citation mark after the quoted text
                if (citation.text) {
                    formatted = formatted.replace(citation.text, `${citation.text}${citationMark}`);
                }
            });
            
            // Add citation references at the end
            const citationRefs = citations.map((citation, index) => {
                return `<div class="citation-ref">
                    <sup>[${index + 1}]</sup> 
                    Chapter ${citation.location.chapter}, p. ${citation.location.page}
                </div>`;
            }).join('');
            
            formatted += `<div class="citations">${citationRefs}</div>`;
        }
        
        return formatted;
    }
    
    displaySystemMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message system';
        messageElement.innerHTML = `
            <div class="message-content">
                <em>${message}</em>
            </div>
        `;
        
        this.messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }
    
    displayErrorMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message error';
        messageElement.innerHTML = `
            <div class="message-content">
                <strong>Error:</strong> ${message}
            </div>
        `;
        
        this.messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }
    
    toggleTypingIndicator(show) {
        if (show) {
            this.typingIndicator.classList.add('active');
        } else {
            this.typingIndicator.classList.remove('active');
        }
    }
    
    scrollToBottom() {
        const container = this.messagesContainer.parentElement;
        container.scrollTop = container.scrollHeight;
    }
    
    // Keepalive mechanism
    startKeepAlive() {
        setInterval(() => {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // Every 30 seconds
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.literaryCompanion = new LiteraryCompanion();
    
    // Add some demo functionality for testing
    if (window.location.hostname === 'localhost') {
        // Auto-fill demo book for testing
        document.getElementById('book-title').value = 'The Brothers Karamazov';
        document.getElementById('book-author').value = 'Fyodor Dostoevsky';
    }
});