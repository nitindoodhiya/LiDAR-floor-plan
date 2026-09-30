
# Multi-Room LiDAR Floor Plan Reconstruction Pipeline

A high-performance Python processing pipeline that ingests raw synchronized camera intrinsics, 6-DoF trajectory odometry, metric depth maps, and per-pixel confidence rasters to generate structured multi-room floor plan JSON specifications.

---

## 📋 Features

* **Parallel Processing**: Multi-threaded and multi-process architecture with live status progress bars (`tqdm`).
* **Memory Optimization (OOM Safe)**: Capped worker allocation and dynamic memory garbage collection designed to prevent system crashes on large capture datasets (10,000+ frames).
* **Trajectory-Based Multi-Room Clustering**: Utilizes spatial density clustering (`DBSCAN`) on camera trajectory waypoints to automatically segment scans into individual rooms.
* **Adjacency Topology Graphing**: Automatically constructs connectivity nodes and doorway transitions between adjacent rooms.
* **Standardized JSON Export**: Outputs schema-compliant multi-room metadata including 95% confidence intervals for floor area, ceiling heights, wall lengths, and opening dimensions.

---

## 📁 Dataset Folder Structure

Organize your input capture directory inside `data/capture_01/` according to the structure below:

```text
LiDAR_floor-plan/
├── data/
│   └── capture_01/                 # Input dataset folder
│       ├── camera_matrix.csv       # Intrinsic parameters (fx, fy, cx, cy)
│       ├── imu.csv                 # Synchronized IMU accelerometer & gyro data
│       ├── odometry.csv            # Camera 6-DoF poses and trajectories (x, y, z)
│       │
│       ├── depth/                  # Raw metric depth images (.png)
│       │   ├── frame_0001.png
│       │   ├── frame_0002.png
│       │   └── ...
│       │
│       └── confidence/             # 8-bit per-pixel confidence maps (.png)
│           ├── frame_0001.png      # (Matches corresponding depth map frame)
│           ├── frame_0002.png
│           └── ...
│
├── process_multi_room_lidar.py     # Main runner script
├── requirements.txt                # Python package dependencies
└── README.md

```

---

## ⚙️ Installation & Setup

### 1. Clone Repository & Create Virtual Environment

```bash
git clone [https://github.com/nitindoodhiya/LiDAR-floor-plan.git](https://github.com/nitindoodhiya/LiDAR-floor-plan.git)
cd LiDAR-floor-plan

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

*Required dependencies (`requirements.txt`):*

```text
numpy
pandas
pillow
scikit-learn
tqdm

```

---

## 🚀 Usage

Execute the main pipeline against your target capture folder:

```bash
python lidar.py

```

### Programmatic Execution

You can import and configure execution parameters directly in your Python code:

```python
from process_multi_room_lidar import process_multi_room_lidar

# Run with custom process worker caps and frame stride settings
result = process_multi_room_lidar(
    capture_dir="data/capture_01",
    max_workers=2,     # Limit worker processes to avoid system memory exhaustion (OOM)
    frame_stride=1     # Process every frame (e.g., set to 2 to process every 2nd frame)
)

```

---

## 📊 Output Schema (`multi_room_floorplan.json`)

The generated JSON file will be written to `data/capture_01/multi_room_floorplan.json`:

```json
{
  "capture_id": "capture_01",
  "tier": "lidar",
  "processing_status": "SUCCESS",
  "total_rooms_detected": 2,
  "metrics": {
    "total_images_found": 9745,
    "images_covered_successfully": 9720,
    "total_3d_points_extracted": 45892010
  },
  "stitched_plan": {
    "total_floor_area_sqm": 42.8,
    "area_confidence_interval": [41.94, 43.66],
    "adjacency_graph": [
      {
        "from": "room_01",
        "to": "room_02",
        "type": "doorway"
      }
    ]
  },
  "rooms": [
    {
      "room_id": "room_01",
      "label": "Room 1",
      "ceiling_height_m": 2.75,
      "height_confidence_interval": [2.74, 2.76],
      "floor_area_sqm": 21.4,
      "walls": [
        {"id": "room_01_w1", "length_m": 4.62, "ci_95": [4.6, 4.64]},
        {"id": "room_01_w2", "length_m": 5.78, "ci_95": [5.76, 5.80]},
        {"id": "room_01_w3", "length_m": 4.62, "ci_95": [4.6, 4.64]},
        {"id": "room_01_w4", "length_m": 5.78, "ci_95": [5.76, 5.80]}
      ],
      "openings": [
        {
          "id": "room_01_door_01",
          "type": "door",
          "width_m": 0.88,
          "ci_95": [0.87, 0.89],
          "wall_id": "room_01_w1"
        }
      ]
    }
  ]
}

```

---

## 🛠️ Memory & Performance Tuning

If you encounter system termination (`Killed`) on lower-RAM hardware:

1. **Reduce Worker Threads**: Pass `max_workers=2` or `max_workers=1` to lower simultaneous peak RAM utilization.
2. **Increase Frame Stride**: Set `frame_stride=2` to process every 2nd frame, cutting processing duration and memory usage in half while maintaining structural coverage.
3. **Adjust Clustering Sensitivity**: Fine-tune `eps` (spatial radius in meters) inside `process_multi_room_lidar.py` under the `DBSCAN` step to tighten or widen room segmentation boundaries.

## Sample Results

Sample results are stored in the main faolder as result_*.json
```