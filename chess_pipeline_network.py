"""
Network-based chess pipeline that accepts images from iPhone via HTTP upload.
Run this script, then use an iPhone app to upload images.

iPhone Apps that work:
- "IP Webcam" (free) - can upload images via HTTP
- "EpocCam" - can stream as webcam
- Or use Safari to upload images via web interface
"""

import cv2
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
from chess_pipeline import ChessPipeline
import tempfile
import shutil

class ChessImageHandler(BaseHTTPRequestHandler):
    pipeline = None
    
    def do_GET(self):
        """Serve a simple HTML page for image upload"""
        if self.path == '/' or self.path == '/upload':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Chess Pipeline - iPhone Upload</title>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body { font-family: Arial; text-align: center; padding: 20px; }
                    input[type="file"] { margin: 20px; font-size: 16px; }
                    button { padding: 10px 20px; font-size: 16px; background: #4CAF50; color: white; border: none; border-radius: 5px; }
                    #status { margin-top: 20px; padding: 10px; }
                </style>
            </head>
            <body>
                <h1>Chess Pipeline - Image Upload</h1>
                <p>Upload a chess board image from your iPhone</p>
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/*" capture="environment" id="fileInput">
                    <br>
                    <button type="submit">Process Image</button>
                </form>
                <div id="status"></div>
                <script>
                    document.getElementById('uploadForm').addEventListener('submit', function(e) {
                        e.preventDefault();
                        var formData = new FormData();
                        var fileInput = document.getElementById('fileInput');
                        if (fileInput.files.length === 0) {
                            document.getElementById('status').innerHTML = '<p style="color: red;">Please select an image first!</p>';
                            return;
                        }
                        formData.append('image', fileInput.files[0]);
                        document.getElementById('status').innerHTML = '<p>Processing...</p>';
                        fetch('/upload', {
                            method: 'POST',
                            body: formData
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                document.getElementById('status').innerHTML = '<p style="color: green;">✓ Image processed successfully!</p>';
                            } else {
                                document.getElementById('status').innerHTML = '<p style="color: red;">Error: ' + data.error + '</p>';
                            }
                        })
                        .catch(error => {
                            document.getElementById('status').innerHTML = '<p style="color: red;">Error: ' + error + '</p>';
                        });
                    });
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle image upload"""
        if self.path == '/upload':
            try:
                content_length = int(self.headers['Content-Length'])
                content_type = self.headers.get('Content-Type', '')
                
                # Read the multipart form data
                post_data = self.rfile.read(content_length)
                
                # Simple multipart parsing (basic implementation)
                # For production, use a proper library like `python-multipart`
                if b'Content-Type: image' in post_data or b'image' in content_type:
                    # Extract image data
                    # Find image data boundaries
                    boundary = content_type.split('boundary=')[-1] if 'boundary=' in content_type else None
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        # Simple extraction - find image data
                        if boundary:
                            parts = post_data.split(b'--' + boundary.encode())
                            for part in parts:
                                if b'Content-Type: image' in part or b'image/jpeg' in part or b'image/png' in part:
                                    # Extract image data (after headers)
                                    image_data = part.split(b'\r\n\r\n', 1)
                                    if len(image_data) > 1:
                                        image_bytes = image_data[1].rstrip(b'--\r\n')
                                        tmp_file.write(image_bytes)
                                        tmp_file.flush()
                                        break
                        else:
                            # Assume raw image data
                            tmp_file.write(post_data)
                            tmp_file.flush()
                        
                        tmp_path = tmp_file.name
                    
                    # Process the image
                    print(f"\nProcessing uploaded image: {tmp_path}")
                    print("-" * 50)
                    
                    frame = cv2.imread(tmp_path)
                    if frame is None:
                        raise Exception("Could not read image file")
                    
                    # Process with pipeline
                    if self.pipeline:
                        out = self.pipeline.process_frame(frame)
                        print("-" * 50)
                        print("Image processed successfully!")
                        
                        # Save processed image
                        processed_path = tmp_path.replace('.jpg', '_processed.jpg')
                        cv2.imwrite(processed_path, out)
                        print(f"Processed image saved to: {processed_path}")
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                    # Send success response
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    response = json.dumps({'success': True, 'message': 'Image processed successfully'})
                    self.wfile.write(response.encode())
                    
                else:
                    raise Exception("No image data found in upload")
                    
            except Exception as e:
                print(f"Error processing upload: {e}")
                import traceback
                traceback.print_exc()
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({'success': False, 'error': str(e)})
                self.wfile.write(response.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

def get_local_ip():
    """Get local IP address"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a remote address (doesn't actually send data)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def main():
    pipeline = ChessPipeline()
    ChessImageHandler.pipeline = pipeline
    
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            print("Usage: python chess_pipeline_network.py [port]")
            return
    
    local_ip = get_local_ip()
    server_address = ('', port)
    httpd = HTTPServer(server_address, ChessImageHandler)
    
    print("=" * 60)
    print("Chess Pipeline - Network Image Server")
    print("=" * 60)
    print(f"\nServer running on:")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}")
    print(f"\nOn your iPhone:")
    print(f"  1. Open Safari")
    print(f"  2. Go to: http://{local_ip}:{port}")
    print(f"  3. Upload chess board images")
    print(f"\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()
        print("Server stopped.")

if __name__ == "__main__":
    main()



