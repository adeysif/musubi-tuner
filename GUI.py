import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import queue
import os
from pathlib import Path

class MusubiTunerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Musubi Tuner GUI")
        self.root.geometry("900x850")

        # --- Style ---
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[10, 5], font=('Segoe UI', 10))
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("Header.TLabel", font=('Segoe UI', 12, 'bold'))
        style.configure("Status.TLabel", font=('Segoe UI', 9))

        # --- Centralized Variable for Dataset Config Path ---
        self.dataset_config_path = tk.StringVar(value="dataset_config.toml")

        # --- Main Layout ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, pady=(0, 10))

        # --- Create Tabs in Workflow Order ---
        self.create_config_tab()
        self.create_tab_cache_latents()
        self.create_tab_cache_text()
        self.create_tab_train()
        
        # --- Output Console ---
        console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding="10")
        console_frame.pack(fill="both", expand=True)
        self.console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, bg="#2b2b2b", fg="#f0f0f0", font=("Consolas", 9))
        self.console.pack(fill="both", expand=True)
        self.output_queue = queue.Queue()
        self.root.after(100, self.process_queue)

    # --- TAB 1: DATASET CONFIG ---
    def create_config_tab(self):
        tab_config = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(tab_config, text="1. Dataset Config")
        ttk.Label(tab_config, text="Edit Dataset Configuration", style="Header.TLabel").grid(row=0, column=0, columnspan=3, pady=(0, 15), sticky="w")
        controls_frame = ttk.Frame(tab_config)
        controls_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)
        ttk.Label(controls_frame, text="Config File:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        entry = ttk.Entry(controls_frame, textvariable=self.dataset_config_path)
        entry.grid(row=0, column=1, sticky="ew")
        browse_btn = ttk.Button(controls_frame, text="Browse...", command=lambda: self.browse_file(self.dataset_config_path))
        browse_btn.grid(row=0, column=2, padx=(5, 10))
        load_btn = ttk.Button(controls_frame, text="Load from File", command=self.load_toml_file)
        load_btn.grid(row=0, column=3, padx=(0, 5))
        save_btn = ttk.Button(controls_frame, text="Save to File", command=self.save_toml_file)
        save_btn.grid(row=0, column=4)
        editor_frame = ttk.LabelFrame(tab_config, text="Editor", padding="5")
        editor_frame.grid(row=2, column=0, columnspan=3, sticky="nsew")
        self.toml_editor = scrolledtext.ScrolledText(editor_frame, wrap=tk.WORD, height=20, font=("Consolas", 10))
        self.toml_editor.pack(fill="both", expand=True)
        default_toml_content = "[general]\nresolution = [512, 512]\ncaption_extension = \".txt\"\nbatch_size = 2\nenable_bucket = true\nbucket_no_upscale = false\n\n[[datasets]]\nimage_directory = \"/path/to/your/images\"\ncache_directory = \"/path/to/your/images/cache\"\nnum_repeats = 1\n"
        self.toml_editor.insert(tk.END, default_toml_content)
        self.config_status_label = ttk.Label(tab_config, text="Ready.", style="Status.TLabel")
        self.config_status_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        tab_config.rowconfigure(2, weight=1)
        tab_config.columnconfigure(0, weight=1)

    # --- TAB 2: CACHE LATENTS ---
    def create_tab_cache_latents(self):
        tab1 = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(tab1, text="2. Cache Latents")
        ttk.Label(tab1, text="Cache Image Latents", style="Header.TLabel").grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky="w")
        self.add_readonly_entry(tab1, "Dataset Config:", self.dataset_config_path, 1)
        self.vae_path_t1 = tk.StringVar(value="/path/to/your/vae.safetensors")
        self.add_file_picker(tab1, "VAE Model:", self.vae_path_t1, 2)
        run_button = ttk.Button(tab1, text="Run Cache Latents", command=self.run_step_cache_latents)
        run_button.grid(row=3, column=0, columnspan=3, pady=(20, 0), sticky="ew")

    # --- TAB 3: CACHE TEXT ENCODER ---
    def create_tab_cache_text(self):
        tab2 = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(tab2, text="3. Cache Text Encoder")
        ttk.Label(tab2, text="Cache Text Encoder Outputs", style="Header.TLabel").grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky="w")
        self.add_readonly_entry(tab2, "Dataset Config:", self.dataset_config_path, 1)
        self.text_encoder_path_t2 = tk.StringVar(value="/path/to/your/text_encoder.safetensors")
        self.batch_size_t2 = tk.StringVar(value="16")
        self.add_file_picker(tab2, "Text Encoder Model:", self.text_encoder_path_t2, 2)
        self.add_entry(tab2, "Batch Size:", self.batch_size_t2, 3)
        run_button = ttk.Button(tab2, text="Run Cache Text Encoder", command=self.run_step_cache_text)
        run_button.grid(row=4, column=0, columnspan=3, pady=(20, 0), sticky="ew")

    # --- TAB 4: TRAIN LORA ---
    def create_tab_train(self):
        tab3 = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab3, text="4. Train LoRA")
        paths_frame = ttk.LabelFrame(tab3, text="Models & Paths", padding=10)
        paths_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        paths_frame.columnconfigure(1, weight=1)
        self.add_readonly_entry(paths_frame, "Dataset Config:", self.dataset_config_path, 0)
        self.dit_path = tk.StringVar(value="/path/to/your/dit_model.safetensors")
        self.vae_path_t3 = tk.StringVar(value="/path/to/your/vae.safetensors")
        self.text_encoder_path_t3 = tk.StringVar(value="/path/to/your/text_encoder.safetensors")
        self.sample_prompts_path = tk.StringVar(value="/path/to/your/prompts.txt")
        self.output_dir = tk.StringVar(value="/path/to/your/output/loras")
        self.output_name = tk.StringVar(value="my-lora-model")
        self.add_file_picker(paths_frame, "DiT Model:", self.dit_path, 1)
        self.add_file_picker(paths_frame, "VAE Model:", self.vae_path_t3, 2)
        self.add_file_picker(paths_frame, "Text Encoder:", self.text_encoder_path_t3, 3)
        self.add_file_picker(paths_frame, "Sample Prompts:", self.sample_prompts_path, 4)
        self.add_directory_picker(paths_frame, "Output Directory:", self.output_dir, 5)
        self.add_entry(paths_frame, "Output Name:", self.output_name, 6)
        params_frame = ttk.LabelFrame(tab3, text="Training Parameters", padding=10)
        params_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.lr = tk.StringVar(value="2e-4")
        self.max_steps = tk.StringVar(value="2500")
        self.save_steps = tk.StringVar(value="100")
        self.sample_steps = tk.StringVar(value="100")
        self.seed = tk.StringVar(value="7626")
        self.network_dim = tk.StringVar(value="16")
        self.network_alpha = tk.StringVar(value="16")
        self.add_entry(params_frame, "Learning Rate:", self.lr, 0)
        self.add_entry(params_frame, "Max Train Steps:", self.max_steps, 1)
        self.add_entry(params_frame, "Save Every N Steps:", self.save_steps, 2)
        self.add_entry(params_frame, "Sample Every N Steps:", self.sample_steps, 3)
        self.add_entry(params_frame, "Network Dim:", self.network_dim, 4)
        self.add_entry(params_frame, "Network Alpha:", self.network_alpha, 5)
        self.add_entry(params_frame, "Seed:", self.seed, 6)
        flags_frame = ttk.LabelFrame(tab3, text="Flags & Options", padding=10)
        flags_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.mixed_precision = tk.StringVar(value="bf16")
        self.optimizer = tk.StringVar(value="adamw8bit")
        self.sdpa = tk.BooleanVar(value=True)
        self.grad_checkpoint = tk.BooleanVar(value=True)
        self.fp8_base = tk.BooleanVar(value=True)
        self.fp8_vl = tk.BooleanVar(value=True)
        ttk.Label(flags_frame, text="Mixed Precision:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(flags_frame, textvariable=self.mixed_precision, values=["no", "fp16", "bf16", "fp8"]).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(flags_frame, text="Optimizer:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Combobox(flags_frame, textvariable=self.optimizer, values=["adamw", "adamw8bit", "prodigy", "sgd", "lion"]).grid(row=0, column=3, sticky="ew", padx=5, pady=2)
        ttk.Checkbutton(flags_frame, text="--sdpa", variable=self.sdpa).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(flags_frame, text="--gradient_checkpointing", variable=self.grad_checkpoint).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(flags_frame, text="--fp8_base", variable=self.fp8_base).grid(row=1, column=2, sticky="w")
        ttk.Checkbutton(flags_frame, text="--fp8_vl", variable=self.fp8_vl).grid(row=1, column=3, sticky="w")
        run_button = ttk.Button(tab3, text="Start LoRA Training", command=self.run_step_train)
        run_button.grid(row=2, column=0, columnspan=2, pady=(20, 0), sticky="ew")
        tab3.columnconfigure(0, weight=1)
        tab3.columnconfigure(1, weight=1)

    # --- WIDGET HELPERS ---
    def add_widget(self, parent, label_text, widget_class, variable, row, **kwargs):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        widget = widget_class(parent, textvariable=variable, **kwargs)
        widget.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        parent.columnconfigure(1, weight=1)
        return widget
    def add_entry(self, parent, label_text, variable, row): return self.add_widget(parent, label_text, ttk.Entry, variable, row)
    def add_readonly_entry(self, parent, label_text, variable, row):
        entry = self.add_widget(parent, label_text, ttk.Entry, variable, row, state="readonly")
        entry.grid(columnspan=2)
        return entry
    def add_file_picker(self, parent, label_text, variable, row):
        self.add_entry(parent, label_text, variable, row)
        ttk.Button(parent, text="Browse...", command=lambda: self.browse_file(variable)).grid(row=row, column=2, sticky="ew", padx=5, pady=2)
    def add_directory_picker(self, parent, label_text, variable, row):
        self.add_entry(parent, label_text, variable, row)
        ttk.Button(parent, text="Browse...", command=lambda: self.browse_directory(variable)).grid(row=row, column=2, sticky="ew", padx=5, pady=2)

    # --- FILE DIALOGS ---
    def browse_file(self, string_var):
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a File", filetypes=(("TOML files", "*.toml"), ("All files", "*.*")))
        if filename: string_var.set(filename)
    def browse_directory(self, string_var, title="Select a Folder"):
        directory = filedialog.askdirectory(initialdir=os.getcwd(), title=title)
        if directory: string_var.set(directory)

    # --- TOML FILE HANDLING ---
    def load_toml_file(self):
        filepath = self.dataset_config_path.get()
        try:
            with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
            self.toml_editor.delete(1.0, tk.END); self.toml_editor.insert(tk.END, content)
            self.config_status_label.config(text=f"✅ Loaded '{os.path.basename(filepath)}'", foreground="green")
        except Exception as e: self.config_status_label.config(text=f"❌ Error: {e}", foreground="red")
    def save_toml_file(self):
        filepath = self.dataset_config_path.get()
        content = self.toml_editor.get(1.0, tk.END)
        try:
            with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
            self.config_status_label.config(text=f"✅ Saved to '{os.path.basename(filepath)}'", foreground="green")
        except Exception as e: self.config_status_label.config(text=f"❌ Error: {e}", foreground="red")

    # --- SUBPROCESS/CONSOLE LOGIC ---
    def process_queue(self):
        try:
            while True:
                item = self.output_queue.get_nowait()
                self.console.insert(tk.END, item)
                self.console.see(tk.END)
        except queue.Empty:
            self.root.after(100, self.process_queue)

    def start_command(self, command):
        threading.Thread(target=self.run_command_thread, args=(command,), daemon=True).start()
        
    def run_command_thread(self, command):
        self.output_queue.put(f"▶️ Running command:\n{' '.join(filter(None, command))}\n\n")
        try:
            final_command = [arg for arg in command if arg]
            process = subprocess.Popen(final_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0, encoding='utf-8', errors='replace')
            for line in iter(process.stdout.readline, ''): self.output_queue.put(line)
            process.wait()
            self.output_queue.put(f"\n✅ --- Process Finished (Exit Code: {process.returncode}) ---\n")
        except Exception as e: self.output_queue.put(f"\n❌ --- ERROR: {e} ---\n")

    # --- RUN STEPS ---
    def run_step_cache_latents(self): self.start_command(["uv", "run", "qwen_image_cache_latents.py", "--dataset_config", self.dataset_config_path.get(), "--vae", self.vae_path_t1.get()])
    def run_step_cache_text(self): self.start_command(["uv", "run", "qwen_image_cache_text_encoder_outputs.py", "--dataset_config", self.dataset_config_path.get(), "--text_encoder", self.text_encoder_path_t2.get(), "--batch_size", self.batch_size_t2.get()])
    def run_step_train(self):
        command = ["uv", "run", "--extra", "cu128", "accelerate", "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", self.mixed_precision.get()]
        command.append("src/musubi_tuner/qwen_image_train_network.py")
        command.extend(["--dit", self.dit_path.get(), "--dataset_config", self.dataset_config_path.get()])
        if self.sdpa.get(): command.append("--sdpa")
        command.extend(["--mixed_precision", self.mixed_precision.get()])
        if self.fp8_base.get(): command.append("--fp8_base")
        command.extend(["--optimizer_type", self.optimizer.get(), "--learning_rate", self.lr.get()])
        if self.sdpa.get(): command.append("--sdpa")
        if self.grad_checkpoint.get(): command.append("--gradient_checkpointing")
        command.extend(["--max_data_loader_n_workers", "2", "--persistent_data_loader_workers", "--network_module", "networks.lora_qwen_image", "--network_dim", self.network_dim.get(), "--network_alpha", self.network_alpha.get(), "--timestep_sampling", "shift", "--discrete_flow_shift", "2.2", "--max_train_steps", self.max_steps.get(), "--save_every_n_steps", self.save_steps.get(), "--seed", self.seed.get(), "--output_dir", self.output_dir.get(), "--output_name", self.output_name.get(), "--vae", self.vae_path_t3.get(), "--text_encoder", self.text_encoder_path_t3.get()])
        if self.fp8_vl.get(): command.append("--fp8_vl")
        if sample_prompts_file := self.sample_prompts_path.get(): command.extend(["--sample_prompts", sample_prompts_file])
        command.extend(["--sample_every_n_steps", self.sample_steps.get(), "--sample_every_n_epoch", "20"])
        self.start_command(command)

if __name__ == "__main__":
    root = tk.Tk()
    app = MusubiTunerGUI(root)
    root.mainloop()