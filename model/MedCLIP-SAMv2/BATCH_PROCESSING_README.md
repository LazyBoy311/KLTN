# MedCLIP-SAMv2 Batch Processing Guide

Hướng dẫn sử dụng batch processing để xử lý nhiều ảnh với MedCLIP-SAMv2 mà không cần load model nhiều lần.

## Tổng quan

Batch processing cho phép bạn:

- Xử lý nhiều ảnh với các prompt đã chuẩn bị sẵn
- Load model chỉ một lần duy nhất
- Theo dõi tiến trình xử lý
- Lưu kết quả và thống kê chi tiết
- Xử lý song song (tùy chọn)

## Cấu trúc thư mục

```
MedCLIP-SAMv2/
├── run.py                    # Script chính (không thay đổi)
├── batch_run.py             # Script batch processing mới
├── create_config.py         # Helper script tạo cấu hình
├── sample_config.json       # File cấu hình mẫu
└── BATCH_PROCESSING_README.md # File này
```

## Cách sử dụng

### 1. Tạo file cấu hình

#### Từ thư mục ảnh:

```bash
# Tạo cấu hình từ thư mục với prompt mặc định
python create_config.py --image-dir ./images --output config.json

# Tạo cấu hình với prompt tùy chỉnh
python create_config.py --image-dir ./images --output config.json --prompt "Brain MRI scan showing tumor"

# Tạo cấu hình với file prompt tùy chỉnh (hỗ trợ nhiều định dạng)
python create_config.py --image-dir ./images --output config.json --custom-prompts prompts.json
python create_config.py --image-dir ./images --output config.json --custom-prompts prompts.csv
python create_config.py --image-dir ./images --output config.json --custom-prompts prompts.txt
```

#### Từ danh sách ảnh:

```bash
python create_config.py --image-list image1.jpg image2.jpg image3.jpg --output config.json
```

#### Tạo file cấu hình mẫu:

```bash
python batch_run.py --create-sample-config
```

### 2. Chạy batch processing

#### Cơ bản:

```bash
python batch_run.py --config config.json --output results/
```

#### Với tùy chọn nâng cao:

```bash
# Sử dụng GPU
python batch_run.py --config config.json --output results/ --device cuda

# Xử lý song song (cẩn thận với GPU)
python batch_run.py --config config.json --output results/ --parallel --max-workers 2

# Lưu kết quả với tên file tùy chỉnh
python batch_run.py --config config.json --output results/ --save-results my_results.json
```

## Định dạng file cấu hình

File cấu hình là JSON với định dạng:

```json
[
  {
    "image_path": "path/to/image1.jpg",
    "text_prompt": "A medical brain MRI scan showing a well-circumscribed, extra-axial mass suggestive of a meningioma tumor."
  },
  {
    "image_path": "path/to/image2.jpg",
    "text_prompt": "A chest X-ray image showing pulmonary nodules in the right lung field."
  }
]
```

## Định dạng file prompt tùy chỉnh

Nếu bạn muốn sử dụng prompt khác nhau cho từng ảnh, hỗ trợ nhiều định dạng file:

### 1. JSON Object Format

```json
{
  "image_name1": "Custom prompt for image 1",
  "image_name2": "Custom prompt for image 2",
  "image_name3": "Custom prompt for image 3"
}
```

### 2. JSON Array Format

```json
[
  {
    "image_name": "image_name1",
    "prompt": "Custom prompt for image 1"
  },
  {
    "image_name": "image_name2",
    "prompt": "Custom prompt for image 2"
  }
]
```

### 3. CSV Format

```csv
image_name,prompt
image_name1,Custom prompt for image 1
image_name2,Custom prompt for image 2
image_name3,Custom prompt for image 3
```

### 4. TXT Format

```txt
image_name1: Custom prompt for image 1
image_name2: Custom prompt for image 2
image_name3: Custom prompt for image 3
```

**Lưu ý**: TXT format hỗ trợ các separator: `:`, `|`, `;`, tab

## Kết quả

### Thư mục output:

```
results/
├── image1/
│   ├── sam_output.png
│   └── postprocessed_map.png
├── image2/
│   ├── sam_output.png
│   └── postprocessed_map.png
└── ...
```

### File kết quả:

- `batch_results.json`: Kết quả chi tiết cho từng ảnh
- `batch_results_summary.json`: Thống kê tổng quan
- `batch_processing.log`: Log file

### Thống kê mẫu:

```
============================================================
BATCH PROCESSING SUMMARY
============================================================
Total Images: 100
Successful: 95
Failed: 5
Success Rate: 95.00%
Total Time: 1800.50s
Average Time per Image: 18.01s
============================================================
```

## Tối ưu hóa hiệu suất

### 1. Sử dụng GPU:

```bash
python batch_run.py --config config.json --output results/ --device cuda
```

### 2. Xử lý song song (CPU):

```bash
python batch_run.py --config config.json --output results/ --parallel --max-workers 4
```

### 3. Giảm logging:

```bash
python batch_run.py --config config.json --output results/ --no-logging
```

## Lưu ý quan trọng

1. **GPU Memory**: Khi sử dụng GPU, đảm bảo có đủ memory cho model
2. **Parallel Processing**: Chỉ sử dụng với CPU hoặc khi có nhiều GPU
3. **File Paths**: Đảm bảo đường dẫn ảnh chính xác
4. **SAM Checkpoint**: Đảm bảo file `sam_vit_h_4b8939.pth` tồn tại

## Troubleshooting

### Lỗi thường gặp:

1. **Model không load được**:

   - Kiểm tra CUDA availability
   - Kiểm tra memory GPU

2. **Ảnh không tìm thấy**:

   - Kiểm tra đường dẫn trong config file
   - Sử dụng đường dẫn tuyệt đối

3. **SAM checkpoint không tìm thấy**:
   - Đảm bảo file `sam_vit_h_4b8939.pth` trong thư mục `segment-anything/`

### Debug:

```bash
# Chạy với logging chi tiết
python batch_run.py --config config.json --output results/ --device cuda
```

## Ví dụ hoàn chỉnh

1. **Chuẩn bị dữ liệu**:

```bash
# Tạo thư mục ảnh
mkdir my_images
# Copy ảnh vào thư mục

# Tạo cấu hình
python create_config.py --image-dir ./my_images --output my_config.json --prompt "Medical image analysis"
```

2. **Chạy batch processing**:

```bash
python batch_run.py --config my_config.json --output my_results/ --device cuda
```

3. **Kiểm tra kết quả**:

```bash
ls my_results/
cat my_results_summary.json
```

## Tích hợp với code hiện tại

Script `run.py` gốc vẫn hoạt động bình thường. Batch processing chỉ là một lớp wrapper tối ưu hóa cho việc xử lý nhiều ảnh.

Bạn có thể sử dụng cả hai cách:

- `run.py` cho xử lý đơn lẻ
- `batch_run.py` cho xử lý hàng loạt
