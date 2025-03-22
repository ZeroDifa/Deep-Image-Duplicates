import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import mimetypes

class FileViewer:
    def __init__(self, file_paths, keep_one=True):
        """
        Initialize the file viewer
        
        Args:
            file_paths (list): List of file paths
            keep_one (bool): If True, keep one file in the list when deleting all files
        """
        self.file_paths = file_paths
        self.keep_one = keep_one
        self.current_index = 0
        self.deleted_files = []
        self.remaining_files = file_paths.copy()
        self.window = None
        self.stop_browsing = False
        self.skip_batch = False
        
    def run(self, title="File Viewer"):
        """Run the file viewer"""
        if not self.remaining_files:
            return [], False
            
        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry("1200x900")
        
        self._init_ui()
        self.show_current_file()
        
        self.window.mainloop()
        return self.remaining_files, self.stop_browsing
        
    def _init_ui(self):
        """Initialize the user interface"""
        self.info_frame = ttk.Frame(self.window, padding=10)
        self.info_frame.pack(fill=tk.X)
        
        ttk.Label(self.info_frame, text="Path:").pack(anchor=tk.W)
        self.path_text = tk.Text(self.info_frame, height=2, wrap=tk.WORD, font=("TkDefaultFont", 10))
        self.path_text.pack(fill=tk.X)
        self.path_text.config(state=tk.DISABLED)
        
        self.size_label = ttk.Label(self.info_frame, text="")
        self.size_label.pack(fill=tk.X)
        
        self.counter_label = ttk.Label(self.info_frame, text="")
        self.counter_label.pack(fill=tk.X)
        
        self.preview_frame = ttk.Frame(self.window)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.preview_label = ttk.Label(self.preview_frame, text="No preview available")
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        self.button_frame = ttk.Frame(self.window, padding=10)
        self.button_frame.pack(fill=tk.X)
        
        self.prev_button = ttk.Button(self.button_frame, text="Prev", command=self.show_prev_file)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(self.button_frame, text="Next", command=self.show_next_file)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            self.button_frame, 
            text="STOP VIEWING", 
            command=self.stop_viewing,
            style="Stop.TButton"
        )
        self.stop_button.pack(side=tk.RIGHT, padx=20)
        
        style = ttk.Style()
        style.configure("Stop.TButton", background="red", foreground="red", font=('Helvetica', 10, 'bold'))
        style.configure("Skip.TButton", font=('Helvetica', 10))
        
        self.delete_all_button = ttk.Button(self.button_frame, text="Delete group", command=self.delete_all_files)
        self.delete_all_button.pack(side=tk.LEFT, padx=5)
        
        self.delete_button = ttk.Button(self.button_frame, text="Delete", command=self.delete_current_file)
        self.delete_button.pack(side=tk.LEFT, padx=5)
        
        self.close_button = ttk.Button(self.button_frame, text="Close", command=self.window.destroy)
        self.close_button.pack(side=tk.LEFT, padx=5)
    
    def stop_viewing(self):
        """Handle stop viewing button click"""
        self.stop_browsing = True
        if messagebox.askyesno("Confirm exit", "Are you sure you want to stop viewing files? All remaining files will be kept."):
            self.window.destroy()
    
    def show_current_file(self):
        """Display the current file"""
        if not self.remaining_files:
            self.window.destroy()
            return
        
        current_file = self.remaining_files[self.current_index]
        
        self.path_text.config(state=tk.NORMAL)
        self.path_text.delete(1.0, tk.END)
        self.path_text.insert(tk.END, current_file)
        self.path_text.config(state=tk.DISABLED)
        
        try:
            file_size = os.path.getsize(current_file)
            size_str = self._format_size(file_size)
            self.size_label.config(text=f"Size: {size_str}")
        except OSError:
            self.size_label.config(text="Size: Reading error")
        
        self.counter_label.config(text=f"File {self.current_index + 1} of {len(self.remaining_files)}")
        
        self._update_preview(current_file)
        
        self._update_delete_button_state()
    
    def _update_preview(self, file_path):
        """Update file preview"""
        self.preview_label.config(image='', text='')
        
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if (mime_type and mime_type.startswith('image/')):
            try:
                image = Image.open(file_path)
                
                max_width = 700*1.5
                max_height = 400*1.5
                width, height = image.size
                
                if width > max_width or height > max_height:
                    ratio = min(max_width/width, max_height/height)
                    width = int(width * ratio)
                    height = int(height * ratio)
                    image = image.resize((width, height), Image.LANCZOS)
                
                photo = ImageTk.PhotoImage(image)
                self.preview_label.config(image=photo)
                self.preview_label.image = photo  # Keep reference
            except Exception as e:
                self.preview_label.config(text=f"Image preview error: {str(e)}")
        elif mime_type and mime_type.startswith('video/'):
            self.preview_label.config(text="[Video file - preview not available]")
        else:
            self.preview_label.config(text=f"[No preview available for this file type: {mime_type or 'unknown type'}]")
    
    def _update_delete_button_state(self):
        """Update delete button state"""
        if self.keep_one and len(self.remaining_files) <= 1:
            self.delete_button.config(state=tk.DISABLED)
            self.delete_all_button.config(state=tk.DISABLED)
        else:
            self.delete_button.config(state=tk.NORMAL)
            self.delete_all_button.config(state=tk.NORMAL)
    
    def show_next_file(self):
        """Show next file"""
        if len(self.remaining_files) > 1:
            self.current_index = (self.current_index + 1) % len(self.remaining_files)
            self.show_current_file()
    
    def show_prev_file(self):
        """Show previous file"""
        if len(self.remaining_files) > 1:
            self.current_index = (self.current_index - 1) % len(self.remaining_files)
            self.show_current_file()
    
    def delete_current_file(self):
        """Delete current file"""
        if not self.remaining_files:
            return
            
        if self.keep_one and len(self.remaining_files) <= 1:
            return
            
        file_to_delete = self.remaining_files[self.current_index]
        
        try:
            os.remove(file_to_delete)
            self.deleted_files.append(file_to_delete)
            
            self.remaining_files.pop(self.current_index)
            
            if self.current_index >= len(self.remaining_files) and self.remaining_files:
                self.current_index = len(self.remaining_files) - 1
            
            if not self.remaining_files or (self.keep_one and len(self.remaining_files) == 1):
                self.window.destroy()
            else:
                self.show_current_file()
        except OSError as e:
            tk.messagebox.showerror("Delete error", f"Could not delete file: {str(e)}")
    
    def delete_all_files(self):
        """Delete all files except one (if keep_one=True)"""
        if not self.remaining_files:
            return
            
        # If keep_one=True, keep one file (the first in the list)
        files_to_delete = self.remaining_files[1:] if self.keep_one else self.remaining_files.copy()
        
        if not files_to_delete:
            return
        
        confirm_msg = "Are you sure you want to delete all files?"
        if self.keep_one:
            confirm_msg = "Are you sure you want to delete all files except the current one?"
        
        if not messagebox.askyesno("Delete confirmation", confirm_msg):
            return
            
        failed_files = []
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                self.deleted_files.append(file_path)
                if file_path in self.remaining_files:
                    self.remaining_files.remove(file_path)
            except OSError as e:
                failed_files.append((file_path, str(e)))
                
        if failed_files:
            error_message = "Could not delete the following files:\n"
            error_message += "\n".join([f"{path}: {error}" for path, error in failed_files])
            messagebox.showerror("Delete error", error_message)
        
        self.current_index = 0
        
        if not self.remaining_files or (self.keep_one and len(self.remaining_files) == 1):
            self.window.destroy()
        else:
            self.show_current_file()
    
    def skip_current_batch(self):
        """Skip current batch of files"""
        if messagebox.askyesno("Skip confirmation", "Are you sure you want to skip this group of files?"):
            self.skip_batch = True
            self.window.destroy()
    
    def _format_size(self, size_bytes):
        """Format file size"""
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.2f} MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.2f} GB"
