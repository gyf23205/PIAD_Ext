import os
import json

import torch
import setting
import numpy as np
import matplotlib
from matplotlib import colors as mcolors
from sklearn.decomposition import PCA
from typing import List


def pad_to_three_dims(array_2d):
    """Pad 2D array columns with zeros until it reaches 3D."""
    if array_2d.shape[1] >= 3:
        return array_2d
    pad_width = 3 - array_2d.shape[1]
    return np.pad(array_2d, ((0, 0), (0, pad_width)), mode='constant')


def embed_full_dataset(model, data_np, device, batch_size=1024):
    """Compute latent embeddings for the entire dataset without stochastic sampling."""
    model.eval()
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    embeddings = []

    with torch.no_grad():
        for start in range(0, data_tensor.shape[0], batch_size):
            end = start + batch_size
            batch = data_tensor[start:end].to(device)
            z, _ = model(batch)
            embeddings.append(z.cpu())

    return torch.cat(embeddings, dim=0).numpy()

def _configure_matplotlib_backend():
    """Ensure an interactive backend is available even in headless sessions."""
    interactive_backends = {
        'qt5agg', 'tkagg', 'macosx', 'wxagg', 'nbagg', 'webagg'
    }
    disable_webagg = os.environ.get('PIAD_DISABLE_WEBAGG', '').lower() in {'1', 'true', 'yes'}
    headless = os.environ.get('DISPLAY') in (None, '')
    requested_backend = os.environ.get('MPLBACKEND', '').lower()
    current_backend = matplotlib.get_backend().lower()

    if not disable_webagg and headless and requested_backend not in interactive_backends and current_backend not in interactive_backends:
        try:
            matplotlib.use('WebAgg')
            print('No GUI display detected; switching Matplotlib backend to WebAgg. '
                  'A browser URL will appear below when the viewer starts.')
        except Exception as exc:
            print(f'Warning: unable to switch to WebAgg backend ({exc}). Plots will be saved only.')

    return matplotlib.get_backend().lower(), headless


ACTIVE_BACKEND, IS_HEADLESS = _configure_matplotlib_backend()
import matplotlib.pyplot as plt

from networks.mlp import MLP_Physical
from datasets.main import load_dataset


def discover_model_paths(root_dir: str, required_file: str = 'model_physical') -> List[str]:
    """Return sorted list of model directories containing the required checkpoint file."""
    if not os.path.isdir(root_dir):
        return []
    paths = []
    # Get all the files whose name contains required_file
    for entry in os.listdir(root_dir):
        if required_file in entry:
            full_path = os.path.join(root_dir, entry)
            paths.append(full_path)
            
    paths.sort()
    return paths


def save_interactive_html(output_path, coords, labels, class_ids, class_names, class_colors_hex, centroid_points):
    """Write a lightweight Plotly HTML viewer without extra Python dependencies."""
    traces = []
    for idx, (name, class_id) in enumerate(zip(class_names, class_ids)):
        mask = labels == class_id
        if not np.any(mask):
            continue
        pts = coords[mask]
        traces.append({
            'type': 'scatter3d',
            'mode': 'markers',
            'name': name,
            'legendgroup': f'class_{class_id}',
            'marker': {'size': 4, 'opacity': 0.65, 'color': class_colors_hex[idx]},
            'x': pts[:, 0].tolist(),
            'y': pts[:, 1].tolist(),
            'z': pts[:, 2].tolist(),
            'hovertemplate': f"{name}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>"
        })

    for centroid in centroid_points:
        coord = centroid['coords']
        traces.append({
            'type': 'scatter3d',
            'mode': 'markers',
            'name': f"{centroid['name']} Centroid",
            'legendgroup': f"centroid_{centroid['label']}",
            'marker': {
                'size': 9,
                'symbol': 'x',
                'line': {'width': 4, 'color': centroid['color']},
                'color': centroid['color']
            },
            'x': [float(coord[0])],
            'y': [float(coord[1])],
            'z': [float(coord[2])],
            'hovertemplate': f"{centroid['name']} Centroid<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>"
        })

    layout = {
        'title': 'Full Dataset Embedding (Interactive)',
        'scene': {
            'xaxis': {'title': 'Component 1'},
            'yaxis': {'title': 'Component 2'},
            'zaxis': {'title': 'Component 3'}
        },
        'legend': {'itemsizing': 'constant'},
        'height': 700,
        'width': 950
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html = f"""<!DOCTYPE html>
<html lang='en'>
  <head>
    <meta charset='utf-8'/>
    <title>Embedding Viewer</title>
    <script src='https://cdn.plot.ly/plotly-2.34.0.min.js'></script>
  </head>
  <body>
    <div id='plot' style='width:100%;height:100%;min-height:640px;'></div>
    <script>
      const data = {json.dumps(traces)};
      const layout = {json.dumps(layout)};
      Plotly.newPlot('plot', data, layout, {{responsive: true}});
    </script>
  </body>
</html>"""
    with open(output_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html)


def save_side_by_side_html(output_path, scenes):
    """Write a Plotly HTML viewer with one 3D scene per model, laid out side by side."""
    if not scenes:
        raise ValueError('No scenes provided for HTML export.')
    traces = []
    layout = {
        'title': 'Full Dataset Embeddings (Interactive, side-by-side)',
        'legend': {'itemsizing': 'constant'},
        'height': 720,
        'width': max(950, 850 * len(scenes)),
        'annotations': []
    }
    total = len(scenes)
    for idx, scene in enumerate(scenes):
        scene_name = 'scene' if idx == 0 else f'scene{idx+1}'
        x0, x1 = idx / total, (idx + 1) / total
        layout[scene_name] = {
            'domain': {'x': [x0, x1], 'y': [0, 1]},
            'xaxis': {'title': 'Component 1'},
            'yaxis': {'title': 'Component 2'},
            'zaxis': {'title': 'Component 3'},
            'title': scene['title']
        }
        layout['annotations'].append({
            'text': scene['title'],
            'xref': 'paper',
            'yref': 'paper',
            'x': (x0 + x1) / 2,
            'y': 1.05,
            'showarrow': False,
            'font': {'size': 14, 'color': '#333'},
            'align': 'center'
        })
        for class_idx, (class_name, class_id) in enumerate(zip(scene['class_names'], scene['class_ids'])):
            mask = scene['labels'] == class_id
            if not np.any(mask):
                continue
            pts = scene['coords'][mask]
            traces.append({
                'type': 'scatter3d',
                'mode': 'markers',
                'name': f"{scene['title']} - {class_name}",
                'legendgroup': f"{scene_name}_class_{int(class_id)}",
                'marker': {'size': 4, 'opacity': 0.65, 'color': scene['class_colors_hex'][class_idx]},
                'x': pts[:, 0].tolist(),
                'y': pts[:, 1].tolist(),
                'z': pts[:, 2].tolist(),
                'hovertemplate': f"{scene['title']} | {class_name}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
                'scene': scene_name
            })
        for centroid in scene['centroids']:
            coord = centroid['coords']
            traces.append({
                'type': 'scatter3d',
                'mode': 'markers',
                'name': f"{scene['title']} - {centroid['name']} Centroid",
                'legendgroup': f"{scene_name}_centroid_{centroid['label']}",
                'marker': {
                    'size': 9,
                    'symbol': 'x',
                    'line': {'width': 4, 'color': centroid['color']},
                    'color': centroid['color']
                },
                'x': [float(coord[0])],
                'y': [float(coord[1])],
                'z': [float(coord[2])],
                'hovertemplate': f"{scene['title']} | {centroid['name']} Centroid<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
                'scene': scene_name
            })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html = f"""<!DOCTYPE html>
<html lang='en'>
  <head>
    <meta charset='utf-8'/>
    <title>Embedding Viewer</title>
    <script src='https://cdn.plot.ly/plotly-2.34.0.min.js'></script>
  </head>
  <body>
    <div id='plot' style='width:100%;height:100%;min-height:640px;'></div>
    <script>
      const data = {json.dumps(traces)};
      const layout = {json.dumps(layout)};
      Plotly.newPlot('plot', data, layout, {{responsive: true}});
    </script>
  </body>
</html>"""
    with open(output_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html)


if __name__ == '__main__':
    # Setup
    dataset_name = 'ALFA'
    data_path = './data'
    ratio_known_outlier = 0.15 #0.3
    ratio_known_normal = 0.15 # 0.2
    ratio_pollution = 0.01 # 0.1
    rko = str(ratio_known_outlier).replace('.','')
    rp = str(ratio_pollution).replace('.','')
    coeff = {
            'sad': 1.0,
            'pred': 4.8,
            'dir': 5.0,
            'cluster': 1.7
        }
    
    model_path = f"./saved_model/physical/model_{rko}_{rp}_{coeff['sad']}_{coeff['pred']}_{coeff['dir']}_{coeff['cluster']}"
    model_path = "." + model_path[1:].replace('.','d')
    model_paths = discover_model_paths(model_path)
    print(f'Found {len(model_paths)} model(s): {", ".join(os.path.basename(p) for p in model_paths)}')
    # setting.init([512, 512, 1024, 2.0])
    setting.init([256, 512, 64, 2.0])
    normal_class = 0
    # known_outlier_class = [1, 2]
    # n_known_outlier_classes = len(known_outlier_class)
    seed = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_samples_per_class = 80
    known_outlier_classes = [1, 3, 4, 6] # One anomaly subtype is know for each kind of anomaly
    n_known_outlier_classes = len(known_outlier_classes)
    subclasses = True
    if subclasses:
        class_all = [0,1,2,3,4,5,6]
    else:
        class_all = [0,1,2,3,4]

    # Order the class by normal, then known anomalies, then unknown anomalies
    class_order = [normal_class] + known_outlier_classes + [cls for cls in class_all if cls not in [normal_class] + known_outlier_classes]

    if subclasses:
        label_anomaly_dict =  { 0: 'Normal',
                                1: 'Engine Failure',
                                2: 'Aileron Failure Right',
                                3: 'Aileron Failure Left',
                                4: 'Elevator Failure',
                                5: 'Rudder Failure Left',
                                6: 'Rudder Failure Right'}
    else:
        label_anomaly_dict =  { 0: 'Normal',
                                1: 'Engine Failure',
                                2: 'Aileron Failure',
                                3: 'Elevator Failure',
                                4: 'Rudder Failure'}

    # Load trained model
    model = MLP_Physical(x_dim=40*47, h_dims=[setting.hd1, setting.hd2], out_dim=47, rep_dim=setting.rep, bias=False)
    # results_dict = torch.load(f'{model_path}/model_physical.tar', map_location=device, weights_only=False)

    # Normal centroid
    # centroids = results_dict['centroids']
    # print(f'Norm of Normal Centroid: {np.linalg.norm(centroids["c_normal"].cpu().numpy()):.4f}')
    # c_normal = centroids['c_normal']
    # for i in range(n_known_outlier_classes):
    #     relative_norm = torch.norm(centroids[f'c_outlier_{i+1}'] - c_normal).item()
    #     print(f'Relative Norm of Anomaly Type {i+1} Centroid to Normal Centroid: {relative_norm:.4f}')
    #     print('---')
    
    # # Relative angle between anomaly centroids
    # for i in range(n_known_outlier_classes):
    #     for j in range(i+1, n_known_outlier_classes):
    #         relative_degree = torch.acos(torch.clamp(torch.dot(centroids[f'c_outlier_{i+1}'], centroids[f'c_outlier_{j+1}']) /
    #                                                (torch.norm(centroids[f'c_outlier_{i+1}']) * torch.norm(centroids[f'c_outlier_{j+1}'])), -1.0, 1.0)).item()
    #         print(f'Angle between Anomaly Type {i+1} and Anomaly Type {j+1} Centroids: {np.degrees(relative_degree):.2f} degrees')
    #         print('---')

    # print('Loading model...')
    # model.load_state_dict(results_dict['net_dict'])
    # model = model.to(device)
    # model.eval()

    print('Loading dataset...')
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_classes, n_known_outlier_classes,
                            ratio_known_normal, ratio_known_outlier, ratio_pollution,
                            random_state=np.random.RandomState(seed),subclasses=subclasses, training=False)

    X_train, y_train, semi_y, X_test, y_test, X_val, y_val = dataset.data_direct()
    X_all = np.concatenate((X_train, X_val, X_test), axis=0)
    y_all = np.concatenate((y_train, y_val, y_test), axis=0)
    labels_all = y_all.astype(int)

    class_ids_train = [normal_class] + list(known_outlier_classes)
    label_to_idx = {label: idx for idx, label in enumerate(class_order)}
    class_names = ['Normal'] + [f'{label_anomaly_dict[label]}' for label in class_order[1:]]
    base_cmap = matplotlib.colormaps.get_cmap('tab10')
    class_colors = base_cmap(np.linspace(0, 1, len(class_order)))
    class_colors_hex = [mcolors.to_hex(color) for color in class_colors]
    centroid_mappings = [('Normal', 'c_normal', class_ids_train[0])]
    centroid_mappings.extend([
        (class_names[idx], f'c_outlier_{idx}', label)
        for idx, label in enumerate(class_ids_train[1:], start=1)
    ])

    # --- New full-dataset visualization (uses every available sample) ---
    # print('Computing embeddings for the entire dataset...')
    # z_all = embed_full_dataset(model, X_all, device)
    # if z_all.shape[1] > 3:
    #     global_pca = PCA(n_components=3)
    #     z_all_projected = global_pca.fit_transform(z_all)
    # else:
    #     global_pca = None
    #     z_all_projected = z_all.copy()

    # z_all_projected = pad_to_three_dims(z_all_projected)
    # labels_all = y_all.astype(int)
    # print(f'Total samples visualized: {labels_all.shape[0]}')
    # for class_name, class_label in zip(class_names, class_all):
    #     print(f'{class_name}: {np.sum(labels_all == class_label)} samples')

    html_scenes = []
    for model_dir in model_paths:
        print(f'\nLoading model from {model_dir}...')
        results_dict = torch.load(model_dir, map_location=device, weights_only=False)
        centroids = results_dict['centroids']
        print(f'Norm of Normal Centroid: {np.linalg.norm(centroids["c_normal"].cpu().numpy()):.4f}')
        for i in range(n_known_outlier_classes):
            relative_norm = torch.norm(centroids[f'c_outlier_{i+1}'] - centroids['c_normal']).item()
            print(f'Relative Norm of Anomaly Type {i+1} Centroid to Normal Centroid: {relative_norm:.4f}')
            print('---')
        for i in range(n_known_outlier_classes):
            for j in range(i+1, n_known_outlier_classes):
                relative_degree = torch.acos(torch.clamp(torch.dot(centroids[f'c_outlier_{i+1}'], centroids[f'c_outlier_{j+1}']) /
                                                      (torch.norm(centroids[f'c_outlier_{i+1}']) * torch.norm(centroids[f'c_outlier_{j+1}'])), -1.0, 1.0)).item()
                print(f'Angle between Anomaly Type {i+1} and Anomaly Type {j+1} Centroids: {np.degrees(relative_degree):.2f} degrees')
                print('---')
        model.load_state_dict(results_dict['net_dict'])
        model = model.to(device)
        model.eval()
        print('Computing embeddings for the entire dataset...')
        z_all = embed_full_dataset(model, X_all, device)
        if z_all.shape[1] > 3:
            global_pca = PCA(n_components=3)
            z_all_projected = global_pca.fit_transform(z_all)
        else:
            global_pca = None
            z_all_projected = z_all.copy()
        z_all_projected = pad_to_three_dims(z_all_projected)
        centroid_points = []
        for centroid_name, centroid_key, centroid_class_label in centroid_mappings:
            centroid_vec = centroids[centroid_key].cpu().numpy().reshape(1, -1)
            if global_pca is not None:
                centroid_projected = global_pca.transform(centroid_vec)
            else:
                centroid_projected = centroid_vec.copy()
                if centroid_projected.shape[1] > 3:
                    centroid_projected = centroid_projected[:, :3]
            centroid_projected = pad_to_three_dims(centroid_projected)[0]
            centroid_points.append({
                'name': centroid_name,
                'label': centroid_class_label,
                'coords': centroid_projected,
                'color': class_colors_hex[label_to_idx[centroid_class_label]]
            })
        fig_full = plt.figure(figsize=(8, 7))
        ax_full = fig_full.add_subplot(111, projection='3d')
        legend_entries_full = set()
        for class_idx, (class_name, class_label) in enumerate(zip(class_names, class_order)):
            mask = np.where(labels_all == class_label)[0]
            if mask.size == 0:
                continue
            label = None
            if class_name not in legend_entries_full:
                label = class_name
                legend_entries_full.add(class_name)
            ax_full.scatter(
                z_all_projected[mask, 0],
                z_all_projected[mask, 1],
                z_all_projected[mask, 2],
                s=10,
                alpha=0.5,
                label=label,
                color=class_colors[class_idx]
            )
        for centroid in centroid_points:
            centroid_label = f"{centroid['name']} Centroid"
            label = None
            if centroid_label not in legend_entries_full:
                label = centroid_label
                legend_entries_full.add(centroid_label)
            ax_full.scatter(
                centroid['coords'][0],
                centroid['coords'][1],
                centroid['coords'][2],
                s=120,
                marker='x',
                linewidths=2,
                color=centroid['color'],
                label=label
            )
        ax_full.set_title(f"{os.path.basename(model_dir)} - Full Dataset Embedding (3D)")
        ax_full.set_xlabel('Component 1')
        ax_full.set_ylabel('Component 2')
        ax_full.set_zlabel('Component 3')
        if legend_entries_full:
            ax_full.legend(loc='upper right', fontsize=9, markerscale=3.0)
        plt.tight_layout()
        # Save static figures for each model
        # full_output_path = f"imgs/{os.path.basename(model_dir)}_embedding_full_dataset_3d.png"
        # fig_full.savefig(full_output_path, dpi=200, bbox_inches='tight')
        # plt.close(fig_full)
        html_scenes.append({
            'title': os.path.basename(model_dir),
            'coords': z_all_projected,
            'labels': labels_all,
            'class_ids': np.array(class_order),
            'class_names': class_names,
            'class_colors_hex': class_colors_hex,
            'centroids': centroid_points
        })
    combined_html_path = 'imgs/embedding_full_dataset_3d_all_models.html'
    save_side_by_side_html(combined_html_path, html_scenes)
    print(f'Interactive HTML viewer saved to {combined_html_path}.')
    print('Done.')