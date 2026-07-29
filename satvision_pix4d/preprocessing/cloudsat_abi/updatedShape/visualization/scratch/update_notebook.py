import json
import re

with open("presentable_vis.ipynb", "r") as f:
    nb = json.load(f)

new_draw_abi = '''def draw_abi_1d(axis, sample, frame, band_index, limits):
    """Draws the 1D ABI sequence as a ribbon matching the CloudSat track."""
    band_1d = sample["ABI/chip"][frame, :, 0, band_index]
    
    distance = track_distance_km(
        sample["CloudSat/latitude"],
        sample["CloudSat/longitude"],
    )
    maximum_distance = max(float(distance[-1]), 1e-6)
    
    band_2d = band_1d[np.newaxis, :]
    lower, upper = limits
    
    axis.imshow(
        band_2d,
        origin="lower",
        aspect="auto",
        extent=[0, maximum_distance, 0, 1],
        cmap='gray',
        vmin=lower,
        vmax=upper
    )
    
    offset = int(sample["ABI/offsets_minutes"][frame])
    scan_time = str(sample["ABI/scan_times"][frame])
    valid = int(sample["ABI/valid_mask"][frame])
    
    axis.set_title(f"ABI C{ABI_BAND:02d} (1D Transect) | {offset:+d} min | valid={valid}\\n{scan_time}")
    axis.set_xlabel("Distance along track (km)")
    axis.set_yticks([])
    axis.set_ylabel("Radiance")
'''

new_make_gif = '''def make_gif(path):
    sample = open_npz_file(path)

    band_index = ABI_BAND - 1
    limits = band_limits(
        sample["ABI/chip"],
        band_index,
    )

    fileName = path.split('/')[-1]

    frames = []
 
    for frame in range(
        sample["ABI/chip"].shape[0]
    ):
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            dpi=90,
            gridspec_kw={'height_ratios': [1, 3]}
        )
 
        draw_abi_1d(
            axes[0],
            sample,
            frame,
            band_index,
            limits
        )
 
        draw_cloudsat_mask(
            axes[1],
            sample,
            variable_name=CLOUDSAT_2D_VAR
        )
 
        figure.suptitle(
            fileName,
            fontsize=10,
        )
 
        figure.tight_layout(
            rect=(0, 0, 1, 0.95)
        )
 
        figure.canvas.draw()
 
        image = PILImage.fromarray(
            np.asarray(
                figure.canvas.buffer_rgba()
            )
        ).convert("RGB")
 
        frames.append(image)
        plt.close(figure)
 
    buffer = BytesIO()
 
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        disposal=2,
        optimize=False,
    )
 
    return buffer.getvalue()
'''

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Remove EAST_LATLONDATA and WEST_LATLONDATA loading blocks
        source = re.sub(r'EAST_LATLONDATA = .*?print\(EAST_abiLat\.shape\)\n', '', source, flags=re.DOTALL)
        source = re.sub(r'WEST_LATLONDATA = .*?print\(WEST_abiLat\.shape\)\n', '', source, flags=re.DOTALL)
        
        # Replace draw_abi
        source = re.sub(r'def draw_abi\(.*?axis\.set_ylabel\("ABI row"\)\n', new_draw_abi + '\n', source, flags=re.DOTALL)
        
        # Replace make_gif
        source = re.sub(r'def make_gif\(path\):.*?return buffer\.getvalue\(\)\n', new_make_gif + '\n', source, flags=re.DOTALL)
        
        # Split source back into lines preserving newlines
        lines = [line + '\n' for line in source.split('\n')]
        if lines:
            lines[-1] = lines[-1][:-1] # remove trailing newline from last element if split added it incorrectly, actually split('\n') is better handled by:
        
        # better split
        import io
        lines = []
        buf = io.StringIO(source)
        for line in buf:
            lines.append(line)
            
        cell["source"] = lines

with open("presentable_vis.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
