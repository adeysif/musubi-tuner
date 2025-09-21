import gradio as gr
import subprocess
import os
import time

os.environ['HF_HUB_CACHE'] = "/workspace/hf_cache"
# ==============================================================================
# ==== 1. COMMAND EXECUTION LOGIC ====
# This is the core function that runs scripts and streams output.
# ==============================================================================
def run_command_stream(command_list, command_name="Command"):
    """
    Runs a command and yields its output line by line.
    Args:
        command_list (list): The command and its arguments as a list of strings.
        command_name (str): A descriptive name for the command being run.
    """
    # Filter out any potential empty strings from the command list
    final_command = [str(arg) for arg in command_list if arg]
    clean_command_str = ' '.join(final_command)

    console_output = f"▶️ Running {command_name}:\n{clean_command_str}\n---------------------------------------\n"
    yield console_output

    try:
        # Using Popen to stream output in real-time
        process = subprocess.Popen(
            final_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1  # Line-buffered
        )

        # Read and yield output line by line
        for line in iter(process.stdout.readline, ''):
            console_output += line
            yield console_output
            time.sleep(0.01) # Small sleep to prevent UI blocking on rapid output

        process.stdout.close()
        process.wait()

        # Final status update
        exit_code = process.returncode
        if exit_code == 0:
            console_output += f"\n---------------------------------------\n✅ Process Finished Successfully (Exit Code: {exit_code})"
        else:
            console_output += f"\n---------------------------------------\n❌ Process Finished with Error (Exit Code: {exit_code})"
        yield console_output

    except FileNotFoundError:
        yield f"❌ ERROR: Command not found. Make sure 'uv' or 'accelerate' is installed and in the system's PATH."
    except Exception as e:
        yield f"❌ An unexpected error occurred:\n{type(e).__name__}: {str(e)}"


# ==============================================================================
# ==== 2. UI LAYOUT AND LOGIC ====
# This section defines the Gradio interface.
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft(), title="Musubi Tuner GUI") as demo:
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                # ----------------------------------------------------------
                # ⚙️ DATASET CONFIG TAB
                # ----------------------------------------------------------
                with gr.Tab("⚙️ Dataset Config"):
                    gr.Markdown("## Edit Dataset Configuration (`.toml`)")
                    
                    with gr.Row():
                        dataset_config_path = gr.Textbox(value="dataset_config.toml", label="Config File Path", scale=3)
                        load_btn = gr.Button("Load from File", scale=1)
                        save_btn = gr.Button("Save to File", scale=1)

                    toml_editor = gr.Textbox(
                        value="""[general]
resolution = [512, 512]
caption_extension = ".txt"
batch_size = 2
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/workspace/data/my_dataset"
cache_directory = "/workspace/data/my_dataset/cache"
num_repeats = 1
""",
                        label="TOML Editor",
                        lines=20,
                        interactive=True
                    )
                    config_status_label = gr.Textbox(label="Status", interactive=False)

                    def load_toml_file(filepath):
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                            status = f"✅ Successfully loaded '{os.path.basename(filepath)}'"
                            return content, status
                        except FileNotFoundError:
                            return "", f"❌ Error: File not found at '{filepath}'"
                        except Exception as e:
                            return "", f"❌ Error loading file: {e}"

                    def save_toml_file(filepath, content):
                        try:
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(content)
                            return f"✅ Successfully saved to '{os.path.basename(filepath)}'"
                        except Exception as e:
                            return f"❌ Error saving file: {e}"

                    load_btn.click(load_toml_file, inputs=[dataset_config_path], outputs=[toml_editor, config_status_label])
                    save_btn.click(save_toml_file, inputs=[dataset_config_path, toml_editor], outputs=[config_status_label])

                # ----------------------------------------------------------
                # QWEN TABS
                # ----------------------------------------------------------
                with gr.Tab("Qwen Training"):
                    with gr.Tabs():
                        with gr.Tab("1. Cache Latents"):
                            gr.Markdown("### Cache Image Latents (Qwen)")
                            qwen_vae_path_t1 = gr.Textbox(value="/workspace/ComfyUI/models/vae/Qwen_VAE.safetensors", label="VAE Model Path")
                            qwen_cache_latents_btn = gr.Button("Run Cache Latents (Qwen)", variant="primary")

                        with gr.Tab("2. Cache Text"):
                            gr.Markdown("### Cache Text Encoder Outputs (Qwen)")
                            qwen_text_encoder_path_t2 = gr.Textbox(value="/workspace/ComfyUI/models/text_encoders/qwen_2.5_vl_7b.safetensors", label="Text Encoder Model Path")
                            qwen_batch_size_t2 = gr.Number(value=16, label="Batch Size", precision=0)
                            qwen_cache_text_btn = gr.Button("Run Cache Text Encoder (Qwen)", variant="primary")
                        
                        with gr.Tab("3. Train LoRA"):
                            gr.Markdown("### Train LoRA (Qwen)")
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("#### Models & Paths")
                                    qwen_dit_path = gr.Textbox(value="/workspace/ComfyUI/models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors", label="DiT Model")
                                    qwen_vae_path_t3 = gr.Textbox(value="/workspace/ComfyUI/models/vae/Qwen_VAE.safetensors", label="VAE Model")
                                    qwen_text_encoder_path_t3 = gr.Textbox(value="/workspace/ComfyUI/models/text_encoders/qwen_2.5_vl_7b.safetensors", label="Text Encoder")
                                    qwen_sample_prompts_path = gr.Textbox(value="/workspace/musubi-tuner/prompts.txt", label="Sample Prompts File")
                                    qwen_output_dir = gr.Textbox(value="/workspace/ComfyUI/models/loras", label="Output Directory")
                                    qwen_output_name = gr.Textbox(value="MyQwenLora", label="Output Name")
                                with gr.Column(scale=1):
                                    gr.Markdown("#### Training Parameters")
                                    qwen_lr = gr.Textbox(value="2e-4", label="Learning Rate")
                                    qwen_max_steps = gr.Number(value=2500, label="Max Train Steps", precision=0)
                                    qwen_save_steps = gr.Number(value=100, label="Save Every N Steps", precision=0)
                                    qwen_sample_steps = gr.Number(value=100, label="Sample Every N Steps", precision=0)
                                    qwen_network_dim = gr.Number(value=16, label="Network Dim", precision=0)
                                    qwen_network_alpha = gr.Number(value=16, label="Network Alpha", precision=0)
                                    qwen_seed = gr.Number(value=7626, label="Seed", precision=0)
                            with gr.Accordion("Flags & Options", open=False):
                                with gr.Row():
                                    qwen_mixed_precision = gr.Dropdown(["no", "fp16", "bf16", "fp8"], value="bf16", label="Mixed Precision")
                                    qwen_optimizer = gr.Dropdown(["adamw", "adamw8bit", "prodigy", "sgd", "lion"], value="adamw8bit", label="Optimizer")
                                with gr.Row():
                                    qwen_sdpa = gr.Checkbox(value=True, label="--sdpa")
                                    qwen_grad_checkpoint = gr.Checkbox(value=True, label="--gradient_checkpointing")
                                    qwen_fp8_base = gr.Checkbox(value=True, label="--fp8_base")
                                    qwen_fp8_vl = gr.Checkbox(value=True, label="--fp8_vl")
                            qwen_train_btn = gr.Button("Start Qwen LoRA Training", variant="primary")
                
                # ----------------------------------------------------------
                # WAN TABS
                # ----------------------------------------------------------
                with gr.Tab("WAN Training"):
                    with gr.Tabs():
                        with gr.Tab("1. Cache Latents"):
                            gr.Markdown("### Cache Image Latents (WAN)")
                            wan_vae_path_t1 = gr.Textbox(value="/workspace/musubi-tuner/models/Wan2_1_VAE_bf16.safetensors", label="VAE Model Path")
                            wan_cache_latents_btn = gr.Button("Run Cache Latents (WAN)", variant="primary")

                        with gr.Tab("2. Cache Text"):
                            gr.Markdown("### Cache Text Encoder Outputs (WAN)")
                            wan_t5_path_t2 = gr.Textbox(value="/workspace/ComfyUI/models/clip/umt5-xxl-enc-fp8_e4m3fn.safetensors", label="T5 Model Path")
                            wan_batch_size_t2 = gr.Number(value=16, label="Batch Size", precision=0)
                            wan_cache_text_btn = gr.Button("Run Cache Text Encoder (WAN)", variant="primary")

                        with gr.Tab("3. Train LoRA"):
                            gr.Markdown("### Train LoRA (WAN)")
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("#### Models & Paths")
                                    wan_dit_path = gr.Textbox(value="/workspace/musubi-tuner/models/wan2.2_t2v_low_noise_14B_fp16.safetensors", label="DiT Model")
                                    wan_dit_high_noise_path = gr.Textbox(value="/workspace/musubi-tuner/models/wan2.2_t2v_high_noise_14B_fp16.safetensors", label="DiT High Noise")
                                    wan_vae_path_t3 = gr.Textbox(value="/workspace/musubi-tuner/models/Wan2_1_VAE_bf16.safetensors", label="VAE Model")
                                    wan_t5_path_t3 = gr.Textbox(value="/workspace/ComfyUI/models/clip/umt5-xxl-enc-fp8_e4m3fn.safetensors", label="T5 Encoder")
                                    wan_sample_prompts_path = gr.Textbox(value="/workspace/musubi-tuner/prompts.txt", label="Sample Prompts File")
                                    wan_output_dir = gr.Textbox(value="/workspace/ComfyUI/models/loras", label="Output Directory")
                                    wan_output_name = gr.Textbox(value="MyWanLora", label="Output Name")
                                with gr.Column(scale=1):
                                    gr.Markdown("#### Training Parameters")
                                    wan_task = gr.Textbox(value="t2v-A14B", label="Task")
                                    wan_timestep_boundary = gr.Number(value=875, label="Timestep Boundary", precision=0)
                                    wan_lr = gr.Textbox(value="2e-5", label="Learning Rate")
                                    wan_max_steps = gr.Number(value=700, label="Max Train Steps", precision=0)
                                    wan_save_steps = gr.Number(value=100, label="Save Every N Steps", precision=0)
                                    wan_sample_steps = gr.Number(value=100, label="Sample Every N Steps", precision=0)
                                    wan_network_dim = gr.Number(value=16, label="Network Dim", precision=0)
                                    wan_network_alpha = gr.Number(value=16, label="Network Alpha", precision=0)
                                    wan_discrete_flow_shift = gr.Textbox(value="1.0", label="Discrete Flow Shift")
                                    wan_seed = gr.Number(value=7626, label="Seed", precision=0)
                            with gr.Accordion("Flags & Options", open=False):
                                with gr.Row():
                                    wan_mixed_precision = gr.Dropdown(["no", "fp16", "bf16", "fp8"], value="bf16", label="Mixed Precision")
                                    wan_optimizer = gr.Dropdown(["adamw", "adamw8bit", "prodigy", "sgd", "lion"], value="adamw8bit", label="Optimizer")
                                with gr.Row():
                                    wan_sdpa = gr.Checkbox(value=True, label="--sdpa")
                                    wan_grad_checkpoint = gr.Checkbox(value=True, label="--gradient_checkpointing")
                                    wan_fp8_base = gr.Checkbox(value=True, label="--fp8_base")
                                    wan_offload_inactive_dit = gr.Checkbox(value=True, label="--offload_inactive_dit")
                                    wan_persistent_workers = gr.Checkbox(value=True, label="--persistent_data_loader_workers")
                            wan_train_btn = gr.Button("Start WAN LoRA Training", variant="primary")
        
        with gr.Column(scale=3):
            console_output = gr.Textbox(
                label="Console Output",
                value="Welcome! Click a run button to start a process.",
                lines=40,
                interactive=False,
                autoscroll=True
            )
                    
    # ==============================================================================
    # ==== 3. COMMAND BUILDERS AND EVENT HANDLERS ====
    # ==============================================================================

    # --- Qwen Event Handlers ---
    def on_qwen_cache_latents(cfg_path, vae_path):
        command = [
            "uv", "run", "qwen_image_cache_latents.py",
            "--dataset_config", cfg_path,
            "--vae", vae_path
        ]
        yield from run_command_stream(command, "Qwen Cache Latents")
    
    qwen_cache_latents_btn.click(
        on_qwen_cache_latents, 
        inputs=[dataset_config_path, qwen_vae_path_t1], 
        outputs=[console_output]
    )

    def on_qwen_cache_text(cfg_path, encoder_path, batch_size):
        command = [
            "uv", "run", "qwen_image_cache_text_encoder_outputs.py",
            "--dataset_config", cfg_path,
            "--text_encoder", encoder_path,
            "--batch_size", batch_size
        ]
        yield from run_command_stream(command, "Qwen Cache Text")

    qwen_cache_text_btn.click(
        on_qwen_cache_text,
        inputs=[dataset_config_path, qwen_text_encoder_path_t2, qwen_batch_size_t2],
        outputs=[console_output]
    )

    def on_qwen_train(*args):
        # Unpack all arguments from the UI
        (cfg_path, dit_path, vae_path, encoder_path, prompts_path, out_dir, out_name,
         lr, max_steps, save_steps, sample_steps, net_dim, net_alpha, seed,
         precision, optimizer, sdpa, grad_ckpt, fp8_base, fp8_vl) = args

        command = [
            "uv", "run", "--extra", "cu128", "accelerate", "launch",
            "--num_cpu_threads_per_process", "1",
            "--mixed_precision", precision,
            "src/musubi_tuner/qwen_image_train_network.py",
            "--dit", dit_path,
            "--dataset_config", cfg_path,
        ]
        if sdpa: command.append("--sdpa")
        command.extend(["--mixed_precision", precision])
        if fp8_base: command.append("--fp8_base")
        command.extend(["--optimizer_type", optimizer, "--learning_rate", lr])
        if grad_ckpt: command.append("--gradient_checkpointing")
        command.extend([
            "--max_data_loader_n_workers", "2",
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_qwen_image",
            "--network_dim", net_dim,
            "--network_alpha", net_alpha,
            "--timestep_sampling", "shift",
            "--discrete_flow_shift", "2.2",
            "--save_state",
            "--max_train_steps", max_steps,
            "--save_every_n_steps", save_steps,
            "--seed", seed,
            "--output_dir", out_dir,
            "--output_name", out_name,
            "--vae", vae_path,
            "--text_encoder", encoder_path,
        ])
        if fp8_vl: command.append("--fp8_vl")
        if prompts_path: command.extend(["--sample_prompts", prompts_path])
        command.extend(["--sample_every_n_steps", sample_steps, "--sample_every_n_epoch", "20"])

        yield from run_command_stream(command, "Qwen LoRA Training")

    qwen_train_btn.click(
        on_qwen_train,
        inputs=[dataset_config_path, qwen_dit_path, qwen_vae_path_t3, qwen_text_encoder_path_t3, 
                qwen_sample_prompts_path, qwen_output_dir, qwen_output_name,
                qwen_lr, qwen_max_steps, qwen_save_steps, qwen_sample_steps, 
                qwen_network_dim, qwen_network_alpha, qwen_seed,
                qwen_mixed_precision, qwen_optimizer, qwen_sdpa, qwen_grad_checkpoint, 
                qwen_fp8_base, qwen_fp8_vl],
        outputs=[console_output]
    )

    # --- WAN Event Handlers ---
    def on_wan_cache_latents(cfg_path, vae_path):
        command = [
            "uv", "run", "wan_cache_latents.py",
            "--dataset_config", cfg_path,
            "--vae", vae_path
        ]
        yield from run_command_stream(command, "WAN Cache Latents")

    wan_cache_latents_btn.click(
        on_wan_cache_latents,
        inputs=[dataset_config_path, wan_vae_path_t1],
        outputs=[console_output]
    )

    def on_wan_cache_text(cfg_path, t5_path, batch_size):
        command = [
            "uv", "run", "wan_cache_text_encoder_outputs.py",
            "--dataset_config", cfg_path,
            "--t5", t5_path,
            "--batch_size", batch_size
        ]
        yield from run_command_stream(command, "WAN Cache Text")

    wan_cache_text_btn.click(
        on_wan_cache_text,
        inputs=[dataset_config_path, wan_t5_path_t2, wan_batch_size_t2],
        outputs=[console_output]
    )
    
    def on_wan_train(*args):
        # Unpack all arguments from the UI
        (cfg_path, dit_path, dit_high_noise_path, vae_path, t5_path, prompts_path,
         out_dir, out_name, task, timestep_boundary, lr, max_steps, save_steps,
         sample_steps, net_dim, net_alpha, flow_shift, seed, precision, optimizer,
         sdpa, grad_ckpt, fp8_base, offload_dit, persistent_workers) = args
        
        command = [
            "uv", "run", "accelerate", "launch",
            "--num_cpu_threads_per_process", "1",
            "--mixed_precision", precision,
            "src/musubi_tuner/wan_train_network.py",
            "--task", task,
            "--dit", dit_path,
            "--dit_high_noise", dit_high_noise_path,
            "--timestep_boundary", timestep_boundary,
            "--dataset_config", cfg_path,
        ]
        if sdpa: command.append("--sdpa")
        command.extend(["--mixed_precision", precision])
        if fp8_base: command.append("--fp8_base")
        command.extend(["--optimizer_type", optimizer])
        if offload_dit: command.append("--offload_inactive_dit")
        command.extend(["--learning_rate", lr])
        if grad_ckpt: command.append("--gradient_checkpointing")
        command.extend(["--max_data_loader_n_workers", "2"])
        if persistent_workers: command.append("--persistent_data_loader_workers")
        command.extend([
            "--network_module", "networks.lora_wan",
            "--network_dim", net_dim,
            "--network_alpha", net_alpha,
            "--timestep_sampling", "shift",
            "--discrete_flow_shift", flow_shift,
            "--max_train_steps", max_steps,
            "--save_every_n_steps", save_steps,
            "--seed", seed,
            "--output_dir", out_dir,
            "--output_name", out_name,
            "--vae", vae_path,
            "--t5", t5_path,
        ])
        if prompts_path: command.extend(["--sample_prompts", prompts_path])
        command.extend(["--sample_every_n_steps", sample_steps, "--sample_at_first"])

        yield from run_command_stream(command, "WAN LoRA Training")

    wan_train_btn.click(
        on_wan_train,
        inputs=[
            dataset_config_path, wan_dit_path, wan_dit_high_noise_path, wan_vae_path_t3,
            wan_t5_path_t3, wan_sample_prompts_path, wan_output_dir, wan_output_name,
            wan_task, wan_timestep_boundary, wan_lr, wan_max_steps, wan_save_steps,
            wan_sample_steps, wan_network_dim, wan_network_alpha, wan_discrete_flow_shift,
            wan_seed, wan_mixed_precision, wan_optimizer, wan_sdpa, wan_grad_checkpoint,
            wan_fp8_base, wan_offload_inactive_dit, wan_persistent_workers
        ],
        outputs=[console_output]
    )

# ==============================================================================
# ==== 4. LAUNCH THE APP ====
# ==============================================================================
if __name__ == "__main__":
    # Launch the app. Use share=True to create a public link if needed.
    # Use server_name="0.0.0.0" to make it accessible on your network.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


